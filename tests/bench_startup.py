"""Measure startup time for each paraketto binary (cold + warm).

Drops page caches between cold runs (requires sudo), then measures warm
runs without cache drops. Reports min/median/max for each.

Usage:
    uv run python tests/bench_startup.py [--runs N] [--binaries BIN,...]
    uv run python tests/bench_startup.py --runs 5
    uv run python tests/bench_startup.py --binaries paraketto.fp8
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

ALL_BINARIES = ["paraketto.cuda", "paraketto.cublas", "paraketto.fp8"]

# Weight files to evict from page cache for cold runs
WEIGHT_FILES = ["paraketto-fp16.bin", "paraketto-fp8.bin"]

STARTUP_RE = re.compile(
    r"startup:\s+(\d+)ms\s+\(cuda=(\d+)\s+prefetch=(\d+)\s+load=(\d+)\)"
)


def find_wav() -> Path:
    for wav in DATA_DIR.rglob("*.wav"):
        return wav
    sys.exit("No WAV files found in data/. Run 'make download-data' first.")


def evict_weight_caches() -> None:
    """Evict weight files from OS page cache using POSIX_FADV_DONTNEED (no sudo)."""
    for name in WEIGHT_FILES:
        path = ROOT / name
        if path.exists():
            fd = os.open(str(path), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def measure_one(binary: Path, wav: Path) -> dict | None:
    result = subprocess.run(
        [str(binary), str(wav)],
        capture_output=True,
        text=True,
    )
    for line in result.stderr.splitlines():
        m = STARTUP_RE.search(line)
        if m:
            return {
                "total":    int(m.group(1)),
                "cuda":     int(m.group(2)),
                "prefetch": int(m.group(3)),
                "load":     int(m.group(4)),
            }
    return None


def fmt(values: list[int]) -> str:
    return f"min={min(values):>4} med={int(median(values)):>4} max={max(values):>4}"


def run_phase(name: str, binaries: list[str], wav: Path, runs: int,
              cold: bool) -> dict[str, dict]:
    """Run measurements, return {binary_name: {totals, cudas, loads}}."""
    results = {}
    for bname in binaries:
        binary = ROOT / bname
        if not binary.exists():
            continue
        totals, cudas, loads = [], [], []
        for i in range(runs):
            if cold:
                evict_weight_caches()
            m = measure_one(binary, wav)
            if m is None:
                print(f"  {bname} {name} {i+1}: no startup line", file=sys.stderr)
                continue
            totals.append(m["total"])
            cudas.append(m["cuda"])
            loads.append(m["load"])
            print(f"  {bname} {name} {i+1}: total={m['total']}ms cuda={m['cuda']} "
                  f"prefetch={m['prefetch']} load={m['load']}",
                  file=sys.stderr)
        results[bname] = {"totals": totals, "cudas": cudas, "loads": loads}
    return results


def print_table(title: str, results: dict[str, dict]) -> None:
    print(f"\n{title}")
    header = f"  {'binary':<20}  {'total (ms)':>28}  {'cuda':>28}  {'load':>28}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bname, data in results.items():
        if data["totals"]:
            print(f"  {bname:<20}  total: {fmt(data['totals'])}"
                  f"  cuda: {fmt(data['cudas'])}"
                  f"  load: {fmt(data['loads'])}")
        else:
            print(f"  {bname:<20}  (not found)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="Runs per binary per phase")
    parser.add_argument("--binaries", type=str, default=None,
                        help="Comma-separated binary names (default: all)")
    args = parser.parse_args()

    binaries = args.binaries.split(",") if args.binaries else ALL_BINARIES
    binaries = [b for b in binaries if (ROOT / b).exists()]
    if not binaries:
        sys.exit("No binaries found. Run 'make paraketto.cuda' etc. first.")

    wav = find_wav()
    print(f"WAV: {wav.relative_to(ROOT)}")
    print(f"Runs: {args.runs} per phase")
    print(f"Binaries: {', '.join(binaries)}")

    # Cold runs first (evict weight files from page cache, no sudo needed)
    cold_results = run_phase("cold", binaries, wav, args.runs, cold=True)
    print_table("COLD (weight files evicted from page cache):", cold_results)

    # Warm runs
    warm_results = run_phase("warm", binaries, wav, args.runs, cold=False)
    print_table("WARM (page cache hot):", warm_results)


if __name__ == "__main__":
    main()
