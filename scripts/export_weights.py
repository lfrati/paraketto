#!/usr/bin/env python3
"""Export ONNX model weights to weights.bin for the CUDA backend.

weights.bin format:
  uint32  magic   = 0x544B5250  ("PRKT" little-endian)
  uint32  version = 2
  [raw FP16 tensor data, 256-byte aligned, in fixed order matching source layout]

The tensor order here matches assign_weight_pointers() in weights.cpp exactly.
This file IS the format specification — no separate index in the binary.
"""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx

REPO_ID       = "istupakov/parakeet-tdt-0.6b-v2-onnx"
MAGIC         = struct.pack("<I", 0x544B5250)  # "PRKT"
VERSION       = struct.pack("<I", 2)
TENSOR_ALIGN  = 256

# ---------------------------------------------------------------------------
# Fixed tensor order — must match assign_weight_pointers() in weights.cpp
# ---------------------------------------------------------------------------

def tensor_order() -> list[str]:
    names = []

    # Subsampling (pre_encode)
    for i in [0, 2, 3, 5, 6]:
        names += [
            f"encoder/pre_encode.conv.{i}.weight",
            f"encoder/pre_encode.conv.{i}.bias",
        ]
    names += [
        "encoder/pre_encode.out.weight",
        "encoder/pre_encode.out.bias",
    ]

    # 24 conformer blocks
    for i in range(24):
        p = f"encoder/layers.{i}"
        names += [
            f"{p}.norm_feed_forward1.weight",
            f"{p}.norm_feed_forward1.bias",
            f"{p}.feed_forward1.linear1.weight",
            f"{p}.feed_forward1.linear2.weight",
            f"{p}.norm_self_att.weight",
            f"{p}.norm_self_att.bias",
            f"{p}.self_attn.linear_q.weight",
            f"{p}.self_attn.linear_k.weight",
            f"{p}.self_attn.linear_v.weight",
            f"{p}.self_attn.linear_pos.weight",
            f"{p}.self_attn.pos_bias_u",
            f"{p}.self_attn.pos_bias_v",
            f"{p}.self_attn.linear_out.weight",
            f"{p}.norm_conv.weight",
            f"{p}.norm_conv.bias",
            f"{p}.conv.pointwise_conv1.weight",
            f"{p}.conv.depthwise_conv.weight",
            f"{p}.conv.depthwise_conv.bias",
            f"{p}.conv.pointwise_conv2.weight",
            f"{p}.norm_feed_forward2.weight",
            f"{p}.norm_feed_forward2.bias",
            f"{p}.feed_forward2.linear1.weight",
            f"{p}.feed_forward2.linear2.weight",
            f"{p}.norm_out.weight",
            f"{p}.norm_out.bias",
        ]

    # Decoder + joint
    names += [
        "decoder/decoder.prediction.embed.weight",
        "decoder/decoder.dec_rnn.lstm.weight_ih",
        "decoder/decoder.dec_rnn.lstm.weight_hh",
        "decoder/decoder.dec_rnn.lstm.bias",
        "decoder/decoder.dec_rnn.lstm.1.weight_ih",
        "decoder/decoder.dec_rnn.lstm.1.weight_hh",
        "decoder/decoder.dec_rnn.lstm.1.bias",
        "decoder/joint.enc.weight",
        "decoder/joint.enc.bias",
        "decoder/joint.pred.weight",
        "decoder/joint.pred.bias",
        "decoder/joint.joint_net.joint_net.2.weight",
        "decoder/joint.joint_net.2.bias",
    ]

    return names


def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def resolve_names(model: onnx.ModelProto) -> dict[str, str]:
    """Map ONNX initializer names to semantic names by tracing the graph."""
    init_names = {init.name for init in model.graph.initializer}
    name_map: dict[str, str] = {}

    for node in model.graph.node:
        init_inputs = [inp for inp in node.input if inp in init_names]
        if not init_inputs:
            continue

        output_name = node.output[0] if node.output else ""
        path = output_name.strip("/")
        parts = path.split("/")
        if parts and "_output_" in parts[-1]:
            last = parts[-1]
            parts = parts[:-1]
            if last.startswith("LSTM_") and not last.startswith("LSTM_output"):
                idx_str = last.split("_")[1]
                parts.append(idx_str)
        semantic_base = ".".join(parts)

        for iname in init_inputs:
            if not iname.startswith("onnx::"):
                name_map[iname] = iname
                continue

            input_list = list(node.input)
            input_idx = input_list.index(iname)

            if node.op_type == "MatMul":
                name_map[iname] = f"{semantic_base}.weight"
            elif node.op_type == "Conv":
                if input_idx == 1:
                    name_map[iname] = f"{semantic_base}.weight"
                elif input_idx == 2:
                    name_map[iname] = f"{semantic_base}.bias"
            elif node.op_type == "LSTM":
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
                name_map[iname] = iname

    for init in model.graph.initializer:
        if init.name not in name_map:
            name_map[init.name] = init.name

    return name_map


def extract_tensors(onnx_path: Path, prefix: str) -> dict[str, np.ndarray]:
    model = onnx.load(str(onnx_path))
    name_map = resolve_names(model)
    tensors = {}
    for init in model.graph.initializer:
        np_dtype = {
            onnx.TensorProto.FLOAT:   np.float32,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.INT32:   np.int32,
            onnx.TensorProto.INT64:   np.int64,
        }.get(init.data_type)
        if np_dtype is None:
            continue
        arr = np.frombuffer(init.raw_data, dtype=np_dtype) if init.raw_data else \
              np.array(init.float_data or init.int32_data or init.int64_data, dtype=np_dtype)
        arr = arr.reshape(list(init.dims))
        tensors[f"{prefix}/{name_map.get(init.name, init.name)}"] = arr
    return tensors


def export_weights(output_path: Path) -> dict[str, np.ndarray]:
    print("Downloading ONNX models...")
    result = subprocess.run(
        ["uvx", "hf", "download", REPO_ID],
        capture_output=True, text=True, check=True,
    )
    onnx_dir = Path(result.stdout.strip())

    all_tensors: dict[str, np.ndarray] = {}
    for name, prefix in [("encoder-model.onnx", "encoder"),
                          ("decoder_joint-model.onnx", "decoder")]:
        path = onnx_dir / name
        if path.exists():
            print(f"Loading {name}...")
            all_tensors.update(extract_tensors(path, prefix))

    # Convert FP32 → FP16, drop int tensors
    for name in list(all_tensors):
        arr = all_tensors[name]
        if arr.dtype in (np.int64, np.int32, np.int8, np.uint8):
            del all_tensors[name]
        elif arr.dtype in (np.float32, np.float64):
            all_tensors[name] = arr.astype(np.float16)

    ordered = tensor_order()

    # Validate all expected tensors are present
    missing = [n for n in ordered if n not in all_tensors]
    if missing:
        print(f"ERROR: {len(missing)} tensors not found in ONNX:")
        for n in missing:
            print(f"  {n}")
        sys.exit(1)

    print(f"\nWriting {output_path} ({len(ordered)} tensors, fixed order)...")
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(VERSION)
        offset = 0
        for name in ordered:
            arr = all_tensors[name]
            raw = arr.astype(np.float16).tobytes()
            # 256-byte align
            pad = align_up(offset, TENSOR_ALIGN) - offset
            if pad:
                f.write(b"\x00" * pad)
                offset += pad
            f.write(raw)
            offset += len(raw)
        # Final alignment
        pad = align_up(offset, TENSOR_ALIGN) - offset
        if pad:
            f.write(b"\x00" * pad)

    size = output_path.stat().st_size
    print(f"Written: {size:,} bytes ({size / 1e6:.1f} MB)")
    return all_tensors


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "weights.bin"
    export_weights(output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
