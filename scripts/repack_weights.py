#!/usr/bin/env python3
"""Convert existing weight files to the current format.

paraketto-fp16.bin: old PRKT v1 (with text header) → new PRKT v2 (headerless, fixed order)
paraketto-fp8.bin:  old PRKTFP8 (24-byte header)  → new PRKTFP8 v1 (16-byte header)

Run once after pulling this update. Does not require ONNX re-export.

Usage:
  uv run python scripts/repack_weights.py              # repack both files in-place
  uv run python scripts/repack_weights.py paraketto-fp16.bin  # repack only paraketto-fp16.bin
"""

import struct
import sys
from pathlib import Path

import numpy as np

MAGIC_FP16  = struct.pack("<I", 0x544B5250)  # "PRKT"
VERSION_FP16 = struct.pack("<I", 2)
TENSOR_ALIGN = 256

MAGIC_FP8    = b"PRKTFP8\x00"
VERSION_FP8  = struct.pack("<I", 1)
PAD_FP8      = struct.pack("<I", 0)


def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


# Fixed tensor order — must match assign_weight_pointers() in weights.cpp
def tensor_order() -> list[str]:
    names = []
    for i in [0, 2, 3, 5, 6]:
        names += [
            f"encoder/pre_encode.conv.{i}.weight",
            f"encoder/pre_encode.conv.{i}.bias",
        ]
    names += [
        "encoder/pre_encode.out.weight",
        "encoder/pre_encode.out.bias",
    ]
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


def repack_fp16(src: Path, dst: Path) -> None:
    print(f"Repacking {src} → {dst}")
    data = src.read_bytes()

    # Parse old PRKT v1 header
    magic, = struct.unpack_from("<I", data, 0)
    version, = struct.unpack_from("<I", data, 4)
    if magic != 0x544B5250:
        print(f"  ERROR: not a PRKT file (magic=0x{magic:08x})")
        sys.exit(1)
    if version == 2:
        print("  Already version 2, nothing to do.")
        return
    if version != 1:
        print(f"  ERROR: unknown version {version}")
        sys.exit(1)

    header_len, = struct.unpack_from("<Q", data, 8)
    header_text = data[16:16 + header_len].decode("utf-8")
    data_start = align_up(16 + header_len, 4096)
    tensor_data = data[data_start:]

    # Parse tensor index
    tensors: dict[str, tuple[int, int, str, tuple]] = {}
    for line in header_text.strip().split("\n"):
        parts = line.split()
        name, offset, size_bytes, dtype = parts[0], int(parts[1]), int(parts[2]), parts[3]
        tensors[name] = (offset, size_bytes, dtype, tuple(int(d) for d in parts[4:]))

    dtype_map = {"fp16": np.float16, "fp32": np.float32}

    # Load all tensors into memory
    loaded: dict[str, bytes] = {}
    for name, (offset, size_bytes, dtype, shape) in tensors.items():
        loaded[name] = tensor_data[offset:offset + size_bytes]

    ordered = tensor_order()
    missing = [n for n in ordered if n not in loaded]
    if missing:
        print(f"  ERROR: {len(missing)} tensors not found:")
        for n in missing[:10]:
            print(f"    {n}")
        sys.exit(1)

    # Write new format
    tmp = dst.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC_FP16)
        f.write(VERSION_FP16)
        offset = 0
        for name in ordered:
            raw = loaded[name]
            pad = align_up(offset, TENSOR_ALIGN) - offset
            if pad:
                f.write(b"\x00" * pad)
                offset += pad
            f.write(raw)
            offset += len(raw)
        pad = align_up(offset, TENSOR_ALIGN) - offset
        if pad:
            f.write(b"\x00" * pad)

    tmp.replace(dst)
    size = dst.stat().st_size
    print(f"  Done: {size:,} bytes ({size / 1e6:.1f} MB)")


def repack_fp8(src: Path, dst: Path) -> None:
    print(f"Repacking {src} → {dst}")
    data = src.read_bytes()

    # Check magic
    if len(data) < 8 or not data[:7] == b"PRKTFP8":
        print(f"  ERROR: not a PRKTFP8 file")
        sys.exit(1)

    # Old format: 24-byte header ("PRKTFP8\0" + uint64 pool_size + uint32 n_scales + uint32 fp16_bytes)
    # New format: 16-byte header ("PRKTFP8\0" + uint32 version + uint32 pad)
    if data[8:16] != b"\x00" * 8:
        # Already new format (has version at byte 8-11, pad at 12-15)
        version_candidate, = struct.unpack_from("<I", data, 8)
        if version_candidate == 1:
            print("  Already version 1 (new format), nothing to do.")
            return

    # Old format: strip the 24-byte header, write new 16-byte header + rest
    payload = data[24:]

    tmp = dst.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC_FP8)
        f.write(VERSION_FP8)
        f.write(PAD_FP8)
        f.write(payload)
    tmp.replace(dst)
    size = dst.stat().st_size
    print(f"  Done: {size:,} bytes ({size / 1e6:.1f} MB)")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["paraketto-fp16.bin", "paraketto-fp8.bin"]

    for name in targets:
        path = repo / name
        if not path.exists():
            print(f"Skipping {name} (not found)")
            continue
        if name.endswith("_fp8.bin") or "fp8" in name:
            repack_fp8(path, path)
        else:
            repack_fp16(path, path)

    print("\nDone. Rebuild all binaries.")


if __name__ == "__main__":
    main()
