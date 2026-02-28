"""Benchmark the C++ parakeet binary: WER and RTFx.

Runs the C++ binary on all utterances from data/{librispeech,earnings22}/manifest.json
in a single invocation, reports per-dataset WER and RTFx, plus long-audio RTFx.

Usage:
    uv run python tests/bench_cpp.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from jiwer import Compose, ReduceToListOfListOfWords, RemovePunctuation, ToLowerCase, wer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BINARY = ROOT / "parakeet"

_normalize = Compose([ToLowerCase(), RemovePunctuation(), ReduceToListOfListOfWords()])

DATASETS = ["librispeech", "earnings22"]
LONG_AUDIO = DATA_DIR / "combined-90s.wav"


def load_manifest(name: str) -> list[dict]:
    manifest_path = DATA_DIR / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        entry["audio_path"] = str(DATA_DIR / name / entry["audio_path"])
    return manifest


def main():
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}", file=sys.stderr)
        print("Run 'make parakeet' first.", file=sys.stderr)
        sys.exit(1)

    # Collect all files across datasets, then run once
    datasets = []
    all_paths = []
    for name in DATASETS:
        manifest = load_manifest(name)
        datasets.append((name, manifest))
        all_paths.extend(e["audio_path"] for e in manifest)

    # Single invocation for all files (first file is warmup internally)
    result = subprocess.run(
        [str(BINARY)] + all_paths,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    hypotheses = result.stdout.strip().split("\n")

    # Parse per-file timing from stderr (format: "6.9s audio, 5.2ms (...), 1327x RTFx")
    stderr_lines = result.stderr.strip().split("\n")
    file_times_ms = []
    for line in stderr_lines:
        m = re.search(r"([\d.]+)ms \(", line)
        if m:
            file_times_ms.append(float(m.group(1)))

    # Split results per dataset
    offset = 0
    for name, manifest in datasets:
        n = len(manifest)
        total_audio = sum(e["duration_s"] for e in manifest)
        references = [e["reference"] for e in manifest]
        hyps = hypotheses[offset:offset + n]
        times = file_times_ms[offset:offset + n]
        offset += n

        wer_pct = wer(
            references, hyps,
            reference_transform=_normalize,
            hypothesis_transform=_normalize,
        ) * 100
        elapsed = sum(times) / 1000
        rtfx = total_audio / elapsed if elapsed > 0 else 0

        print(f"{name}: WER={wer_pct:.2f}% RTFx={rtfx:.0f}x ({n} utts, {total_audio:.0f}s audio)")

    total_audio = sum(e["duration_s"] for m in [d[1] for d in datasets] for e in m)
    total_elapsed = sum(file_times_ms) / 1000
    print(f"total: {total_audio:.0f}s audio in {total_elapsed*1000:.0f}ms, {total_audio/total_elapsed:.0f}x RTFx")

    # Long-audio RTFx (separate invocation — includes its own warmup)
    result = subprocess.run(
        [str(BINARY), str(LONG_AUDIO)],
        capture_output=True, text=True,
    )
    stderr = result.stderr.strip().split("\n")[-1]
    print(f"long-audio: {stderr}")


if __name__ == "__main__":
    main()
