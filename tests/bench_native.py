"""Benchmark a paraketto native binary: WER and RTFx.

Usage:
    uv run python tests/bench_native.py paraketto.cuda
    uv run python tests/bench_native.py paraketto.cublas
    uv run python tests/bench_native.py paraketto.fp8
"""

import sys
from pathlib import Path
from bench_common import bench_server

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bench_native.py <binary>", file=sys.stderr)
        sys.exit(1)
    binary = ROOT / sys.argv[1]
    bench_server(binary, sys.argv[1])
