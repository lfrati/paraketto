"""Benchmark a paraketto CUDA binary: WER and RTFx.

Usage:
    uv run python tests/bench_cuda.py                  # default: paraketto.cuda
    uv run python tests/bench_cuda.py paraketto.cublas  # cuBLAS backend
"""

import sys
from pathlib import Path

from bench_common import bench_server

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    binary = sys.argv[1] if len(sys.argv) > 1 else "paraketto.cuda"
    bench_server(ROOT / binary, binary)
