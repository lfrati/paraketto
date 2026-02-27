"""Benchmark Parakeet TDT V2 TensorRT: WER and RTFx.

Runs all utterances from data/{librispeech,earnings22}/manifest.json,
reports per-dataset WER and RTFx, plus a long-audio RTFx measurement
on data/combined-90s.wav.

Usage:
    uv run python tests/bench.py
"""

import json
import sys
import time
from pathlib import Path

import soundfile as sf
from jiwer import Compose, ReduceToListOfListOfWords, RemovePunctuation, ToLowerCase, wer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"

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
    import parakeet_trt

    print("Loading model...")
    t0 = time.monotonic()
    parakeet_trt.load_model()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s\n")

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

        print(f"{dataset}: WER={wer_pct:.2f}% RTFx={rtfx:.0f}x "
              f"({len(manifest)} utts, {total_audio:.0f}s audio)")

    # Long-audio RTFx (no ground truth, just speed)
    audio, sr = sf.read(str(LONG_AUDIO), dtype="float32")
    audio_dur = len(audio) / sr

    parakeet_trt.transcribe(audio, sr)  # warmup

    runs = 5
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        parakeet_trt.transcribe(audio, sr)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    mean_t = sum(times) / len(times)
    rtfx_long = audio_dur / mean_t
    print(f"long-audio: RTFx={rtfx_long:.0f}x "
          f"({audio_dur:.0f}s audio, {mean_t*1000:.0f}ms mean, {runs} runs)")


if __name__ == "__main__":
    main()
