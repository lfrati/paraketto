#!/usr/bin/env python3
"""Export ONNX model weights to a flat binary file for the CUDA backend.

Resolves anonymous ONNX initializer names (onnx::MatMul_XXXX, onnx::Conv_XXXX,
onnx::LSTM_XXX) to semantic names by tracing the graph structure.

Weight file format (weights.bin):
  [4 bytes: "PRKT" magic]
  [4 bytes: version uint32 = 1]
  [8 bytes: header_len uint64 — byte length of the text index]
  [header_len bytes: text index, one line per tensor]
  [padding to 4096-byte alignment]
  [raw tensor data, each tensor 256-byte aligned]

Header text format (one line per tensor):
  name offset_from_data_start size_bytes dtype dim0 dim1 ...
"""

import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import snapshot_download

REPO_ID = "istupakov/parakeet-tdt-0.6b-v2-onnx"
MAGIC = b"PRKT"
VERSION = 1
HEADER_ALIGN = 4096
TENSOR_ALIGN = 256

ONNX_DTYPE_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.DOUBLE: np.float64,
}

DTYPE_NAMES = {
    np.dtype(np.float16): "fp16",
    np.dtype(np.float32): "fp32",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
}


def align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) & ~(alignment - 1)


def resolve_names(model: onnx.ModelProto) -> dict[str, str]:
    """Walk the ONNX graph and build a mapping from initializer name to semantic name.

    For named initializers (e.g. 'layers.0.norm_conv.weight'), keeps the name as-is.
    For anonymous initializers (e.g. 'onnx::MatMul_6382'), derives the name from
    the consumer node's output path.

    Node output names follow the pattern: /path/to/op/OpType_output_0
    We extract the path component to form the semantic name.
    """
    init_names = {init.name for init in model.graph.initializer}
    name_map: dict[str, str] = {}

    for node in model.graph.node:
        init_inputs = [inp for inp in node.input if inp in init_names]
        if not init_inputs:
            continue

        # Extract semantic path from the first output name
        # e.g. '/layers.0/feed_forward1/linear1/MatMul_output_0' -> 'layers.0.feed_forward1.linear1'
        output_name = node.output[0] if node.output else ""
        path = output_name.strip("/")
        # Remove the trailing '/OpType_output_0' suffix
        parts = path.split("/")
        # Drop the last part if it contains '_output_'
        if parts and "_output_" in parts[-1]:
            # For LSTM_1_output_0, extract the layer suffix before dropping
            last = parts[-1]
            parts = parts[:-1]
            # Handle multi-layer LSTM: output names like LSTM_1_output_0 → append ".1"
            if last.startswith("LSTM_") and not last.startswith("LSTM_output"):
                # Extract layer index from e.g. "LSTM_1_output_0"
                idx_str = last.split("_")[1]
                parts.append(idx_str)
        semantic_base = ".".join(parts)

        for i, iname in enumerate(init_inputs):
            if not iname.startswith("onnx::"):
                # Already has a good name — keep it
                name_map[iname] = iname
                continue

            input_list = list(node.input)
            input_idx = input_list.index(iname)

            # Derive semantic name based on op type and input position
            if node.op_type == "MatMul":
                name_map[iname] = f"{semantic_base}.weight"
            elif node.op_type == "Conv":
                # Conv inputs: [X, W, B]
                if input_idx == 1:
                    name_map[iname] = f"{semantic_base}.weight"
                elif input_idx == 2:
                    name_map[iname] = f"{semantic_base}.bias"
            elif node.op_type == "LSTM":
                # ONNX LSTM inputs: [X, W, R, B, ...]
                if input_idx == 1:
                    name_map[iname] = f"{semantic_base}.weight_ih"
                elif input_idx == 2:
                    name_map[iname] = f"{semantic_base}.weight_hh"
                elif input_idx == 3:
                    name_map[iname] = f"{semantic_base}.bias"
            elif node.op_type == "Add":
                name_map[iname] = f"{semantic_base}.bias"
            elif node.op_type == "Gemm":
                if input_idx == 1:
                    name_map[iname] = f"{semantic_base}.weight"
                elif input_idx == 2:
                    name_map[iname] = f"{semantic_base}.bias"
            else:
                # Unknown op — use raw name
                name_map[iname] = iname

    # Any initializer not consumed by a node (shouldn't happen, but be safe)
    for init in model.graph.initializer:
        if init.name not in name_map:
            name_map[init.name] = init.name

    return name_map


def extract_tensors(onnx_path: Path, prefix: str) -> dict[str, np.ndarray]:
    """Extract all initializer tensors with resolved semantic names."""
    model = onnx.load(str(onnx_path))
    name_map = resolve_names(model)

    tensors = {}
    skipped = 0
    for init in model.graph.initializer:
        np_dtype = ONNX_DTYPE_TO_NP.get(init.data_type)
        if np_dtype is None:
            print(f"  WARNING: skipping {init.name} with unsupported dtype {init.data_type}")
            skipped += 1
            continue

        arr = np.frombuffer(init.raw_data, dtype=np_dtype) if init.raw_data else \
              np.array(init.float_data or init.int32_data or init.int64_data, dtype=np_dtype)
        arr = arr.reshape(list(init.dims))

        semantic_name = name_map.get(init.name, init.name)
        full_name = f"{prefix}/{semantic_name}"

        tensors[full_name] = arr

    if skipped:
        print(f"  Skipped {skipped} tensors with unsupported dtypes")

    return tensors


def export_weights(output_path: Path) -> dict[str, np.ndarray]:
    """Export all ONNX weights to a flat binary file. Returns the tensor dict for verification."""
    print("Downloading ONNX models...")
    onnx_dir = Path(snapshot_download(REPO_ID))
    print(f"ONNX dir: {onnx_dir}")

    # Extract tensors from both FP32 models
    all_tensors: dict[str, np.ndarray] = {}

    encoder_path = onnx_dir / "encoder-model.onnx"
    decoder_path = onnx_dir / "decoder_joint-model.onnx"

    if encoder_path.exists():
        print(f"\nLoading {encoder_path.name}...")
        enc_tensors = extract_tensors(encoder_path, "encoder")
        print(f"  {len(enc_tensors)} tensors extracted")
        all_tensors.update(enc_tensors)

    if decoder_path.exists():
        print(f"\nLoading {decoder_path.name}...")
        dec_tensors = extract_tensors(decoder_path, "decoder")
        print(f"  {len(dec_tensors)} tensors extracted")
        all_tensors.update(dec_tensors)

    # Print resolved names for verification
    print(f"\nResolved tensor names:")
    for name in sorted(all_tensors.keys()):
        arr = all_tensors[name]
        print(f"  {name:70s} {str(list(arr.shape)):20s} {arr.dtype}")

    # Convert all FP32 tensors to FP16 (skip int tensors used for reshape shapes)
    print(f"\nConverting FP32 -> FP16...")
    converted = 0
    dropped = 0
    for name, arr in list(all_tensors.items()):
        if arr.dtype in (np.int64, np.int32, np.int8, np.uint8):
            # Skip shape/metadata tensors
            del all_tensors[name]
            dropped += 1
            continue
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            all_tensors[name] = arr.astype(np.float16)
            converted += 1
    print(f"  Converted {converted} tensors to FP16, dropped {dropped} int tensors")

    # Sort tensors by name for deterministic output
    sorted_names = sorted(all_tensors.keys())

    # Build the data section: pack tensors with 256-byte alignment
    data_parts: list[bytes] = []
    tensor_index: list[tuple[str, int, int, str, tuple]] = []
    current_offset = 0

    for name in sorted_names:
        arr = all_tensors[name]
        raw = arr.tobytes()
        size = len(raw)
        dtype_name = DTYPE_NAMES.get(arr.dtype, str(arr.dtype))

        tensor_index.append((name, current_offset, size, dtype_name, tuple(arr.shape)))
        data_parts.append(raw)

        # Pad to next 256-byte boundary
        padded_size = align_up(size, TENSOR_ALIGN)
        if padded_size > size:
            data_parts.append(b"\x00" * (padded_size - size))
        current_offset += padded_size

    # Build text header
    header_lines = []
    for name, offset, size, dtype_name, shape in tensor_index:
        dims = " ".join(str(d) for d in shape)
        header_lines.append(f"{name} {offset} {size} {dtype_name} {dims}")
    header_text = "\n".join(header_lines).encode("utf-8")

    # Write the file
    print(f"\nWriting {output_path}...")
    with open(output_path, "wb") as f:
        # Magic + version + header_len
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(header_text)))
        f.write(header_text)

        # Pad to 4096-byte alignment
        current_pos = 4 + 4 + 8 + len(header_text)
        pad_to = align_up(current_pos, HEADER_ALIGN)
        if pad_to > current_pos:
            f.write(b"\x00" * (pad_to - current_pos))

        # Write all tensor data
        for part in data_parts:
            f.write(part)

    file_size = output_path.stat().st_size
    print(f"  Written: {file_size:,} bytes ({file_size / 1e6:.1f} MB)")

    # Summary
    print(f"\nSummary:")
    print(f"  Total tensors: {len(sorted_names)}")
    print(f"  Data size: {current_offset:,} bytes ({current_offset / 1e6:.1f} MB)")
    print(f"  File size: {file_size:,} bytes ({file_size / 1e6:.1f} MB)")

    # Per-section breakdown
    for prefix in ["encoder", "decoder"]:
        section = [(n, all_tensors[n]) for n in sorted_names if n.startswith(prefix + "/")]
        total = sum(a.nbytes for _, a in section)
        print(f"  {prefix}: {len(section)} tensors, {total:,} bytes ({total / 1e6:.1f} MB)")

    return all_tensors


def verify_weights(output_path: Path, original_tensors: dict[str, np.ndarray]) -> None:
    """Reload the .bin file and verify every tensor matches the original exactly."""
    print(f"\nVerifying {output_path}...")

    with open(output_path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        assert version == VERSION, f"Bad version: {version}"
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_text = f.read(header_len).decode("utf-8")

        # Skip to data start (4096-byte aligned)
        header_end = 4 + 4 + 8 + header_len
        data_start = align_up(header_end, HEADER_ALIGN)
        f.seek(data_start)
        data = f.read()

    # Parse header
    errors = 0
    checked = 0
    for line in header_text.strip().split("\n"):
        parts = line.split()
        name = parts[0]
        offset = int(parts[1])
        size = int(parts[2])
        dtype_str = parts[3]
        shape = tuple(int(d) for d in parts[4:])

        # Map dtype string back to numpy dtype
        dtype_map = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32, "int64": np.int64}
        np_dtype = dtype_map.get(dtype_str)
        if np_dtype is None:
            print(f"  WARNING: unknown dtype {dtype_str} for {name}")
            continue

        # Extract tensor from data
        raw = data[offset:offset + size]
        loaded = np.frombuffer(raw, dtype=np_dtype).reshape(shape)

        # Compare with original (already converted to FP16)
        original = original_tensors.get(name)
        if original is None:
            print(f"  ERROR: {name} not in original tensors")
            errors += 1
            continue

        if not np.array_equal(loaded, original):
            max_diff = np.max(np.abs(loaded.astype(np.float32) - original.astype(np.float32)))
            print(f"  ERROR: {name} mismatch, max_diff={max_diff}")
            errors += 1
        checked += 1

    if errors:
        print(f"  FAILED: {errors} errors out of {checked} tensors")
        sys.exit(1)
    else:
        print(f"  OK: all {checked} tensors match exactly")


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "weights.bin"
    original_tensors = export_weights(output_path)
    verify_weights(output_path, original_tensors)
    print("\nDone.")


if __name__ == "__main__":
    main()
