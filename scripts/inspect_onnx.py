#!/usr/bin/env python3
"""Inspect ONNX models for parakeet-tdt-0.6b-v2: print all initializer tensors."""

from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import snapshot_download

REPO_ID = "istupakov/parakeet-tdt-0.6b-v2-onnx"

DTYPE_MAP = {
    onnx.TensorProto.FLOAT: ("float32", 4),
    onnx.TensorProto.FLOAT16: ("float16", 2),
    onnx.TensorProto.INT32: ("int32", 4),
    onnx.TensorProto.INT64: ("int64", 8),
    onnx.TensorProto.INT8: ("int8", 1),
    onnx.TensorProto.UINT8: ("uint8", 1),
    onnx.TensorProto.DOUBLE: ("float64", 8),
}


def inspect_model(onnx_path: Path, label: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {label}: {onnx_path.name}")
    print(f"{'=' * 80}")

    model = onnx.load(str(onnx_path))
    graph = model.graph

    # Print inputs/outputs
    print(f"\n  Inputs:")
    for inp in graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param or "?")
        print(f"    {inp.name:40s}  {shape}")

    print(f"\n  Outputs:")
    for out in graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param or "?")
        print(f"    {out.name:40s}  {shape}")

    # Print all initializer tensors
    print(f"\n  Initializers ({len(graph.initializer)} tensors):")
    print(f"  {'Name':60s} {'Shape':25s} {'Dtype':8s} {'Size':>12s}")
    print(f"  {'-'*60} {'-'*25} {'-'*8} {'-'*12}")

    total_bytes = 0
    total_fp16_bytes = 0
    for init in sorted(graph.initializer, key=lambda x: x.name):
        shape = list(init.dims)
        dtype_name, elem_size = DTYPE_MAP.get(init.data_type, (f"dt{init.data_type}", 0))
        numel = 1
        for d in shape:
            numel *= d
        size_bytes = numel * elem_size
        fp16_bytes = numel * 2
        total_bytes += size_bytes
        total_fp16_bytes += fp16_bytes

        shape_str = str(shape)
        print(f"  {init.name:60s} {shape_str:25s} {dtype_name:8s} {size_bytes:>12,}")

    print(f"\n  Total: {len(graph.initializer)} tensors, "
          f"{total_bytes:,} bytes ({total_bytes / 1e6:.1f} MB) FP32, "
          f"{total_fp16_bytes:,} bytes ({total_fp16_bytes / 1e6:.1f} MB) FP16")


def main() -> None:
    print("Downloading ONNX models...")
    onnx_dir = Path(snapshot_download(REPO_ID))
    print(f"ONNX dir: {onnx_dir}")

    # List ONNX files
    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    print(f"Found {len(onnx_files)} ONNX files: {[f.name for f in onnx_files]}")

    for onnx_file in onnx_files:
        label = onnx_file.stem.replace("-model", "").replace("_", " ").title()
        inspect_model(onnx_file, label)


if __name__ == "__main__":
    main()
