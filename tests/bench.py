"""Benchmark Parakeet TDT V2 TensorRT: WER and RTFx.

Runs all utterances from data/{librispeech,earnings22}/manifest.json,
reports per-dataset WER and aggregate RTFx.

Usage:
    uv run python tests/bench.py
"""

import json
import sys
import time
from pathlib import Path

from jiwer import Compose, ReduceToListOfListOfWords, RemovePunctuation, ToLowerCase, wer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"

_normalize = Compose([ToLowerCase(), RemovePunctuation(), ReduceToListOfListOfWords()])

DATASETS = ["librispeech", "earnings22"]


def load_manifest(name: str) -> list[dict]:
    manifest_path = DATA_DIR / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        entry["audio_path"] = str(DATA_DIR / name / entry["audio_path"])
    return manifest


def main():
    import parakeet_trt

    print("Loading model...")
    t0 = time.monotonic()
    parakeet_trt.load_model()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s\n")

    total_audio_all = 0.0
    total_inference_all = 0.0
    all_ok = True

    for dataset in DATASETS:
        manifest = load_manifest(dataset)
        total_audio = sum(e["duration_s"] for e in manifest)

        references = []
        hypotheses = []
        total_inference = 0.0

        for entry in manifest:
            t0 = time.perf_counter()
            hyp = parakeet_trt.transcribe_file(entry["audio_path"])
            elapsed = time.perf_counter() - t0

            references.append(entry["reference"])
            hypotheses.append(hyp)
            total_inference += elapsed

        wer_pct = wer(
            references, hypotheses,
            reference_transform=_normalize,
            hypothesis_transform=_normalize,
        ) * 100

        rtfx = total_audio / total_inference if total_inference > 0 else 0

        total_audio_all += total_audio
        total_inference_all += total_inference

        status = "PASS" if wer_pct < 20 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"{dataset:>12s}:  WER={wer_pct:5.2f}%  RTFx={rtfx:7.1f}x  "
              f"({len(manifest)} utts, {total_audio:.1f}s audio)  [{status}]")

    rtfx_all = total_audio_all / total_inference_all if total_inference_all > 0 else 0
    print(f"\n{'aggregate':>12s}:  RTFx={rtfx_all:7.1f}x  "
          f"({total_audio_all:.1f}s audio in {total_inference_all:.2f}s)")

    if rtfx_all < 400:
        print(f"\nFAIL: aggregate RTFx {rtfx_all:.1f}x < 400x")
        all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
