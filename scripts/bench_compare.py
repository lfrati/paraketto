#!/usr/bin/env python3
"""Compare parakeet (TRT) vs parakeet_cuda transcription and performance."""
import subprocess, sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRT_BIN = ROOT / "parakeet"
CUDA_BIN = ROOT / "parakeet_cuda"

wavs = sorted((ROOT / "data" / "librispeech").glob("*.wav"))
# Exclude combined-60s.wav from text comparison (it's not a test utterance)
test_wavs = [w for w in wavs if not w.name.startswith("combined")]
print(f"Comparing {len(test_wavs)} files (text) + {len(wavs)} files (perf)...\n")

matches = 0
diffs = 0
trt_total_audio = trt_total_inf = trt_total_enc = 0.0
cuda_total_audio = cuda_total_inf = cuda_total_enc = 0.0

def get_text(combined_output):
    """Get transcription text (last non-empty line that doesn't contain RTFx)."""
    lines = combined_output.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and 'RTFx' not in line and 'init:' not in line and 'startup:' not in line:
            return line
    return ""

def get_timing(combined_output):
    m = re.search(r'([\d.]+)s audio, ([\d.]+)ms \(mel=([\d.]+) enc=([\d.]+) dec=([\d.]+)\), ([\d.]+)x RTFx', combined_output)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(4))
    return None, None, None

for wav in wavs:
    trt_out = subprocess.run([str(TRT_BIN), str(wav)], capture_output=True, text=True)
    cuda_out = subprocess.run([str(CUDA_BIN), str(wav)], capture_output=True, text=True)

    trt_combined = trt_out.stdout + trt_out.stderr
    cuda_combined = cuda_out.stdout + cuda_out.stderr

    # Performance
    ta, ti, te = get_timing(trt_combined)
    ca, ci, ce = get_timing(cuda_combined)
    if ta: trt_total_audio += ta; trt_total_inf += ti; trt_total_enc += te
    if ca: cuda_total_audio += ca; cuda_total_inf += ci; cuda_total_enc += ce

    # Text comparison (skip combined files)
    if wav in test_wavs:
        trt_text = get_text(trt_combined)
        cuda_text = get_text(cuda_combined)
        if trt_text == cuda_text:
            matches += 1
        else:
            diffs += 1
            print(f"DIFFER: {wav.name}")
            print(f"  TRT:  {trt_text[:100]}")
            print(f"  CUDA: {cuda_text[:100]}")

n = len(wavs)
print(f"\n=== Text Comparison ===")
print(f"Match: {matches}/{matches+diffs}")

print(f"\n=== TRT Performance ({n} files) ===")
print(f"Total audio: {trt_total_audio:.1f}s, Total inference: {trt_total_inf:.1f}ms")
print(f"Avg enc_ms: {trt_total_enc/n:.1f}, Overall RTFx: {trt_total_audio*1000/trt_total_inf:.0f}x")

print(f"\n=== CUDA Performance ({n} files) ===")
print(f"Total audio: {cuda_total_audio:.1f}s, Total inference: {cuda_total_inf:.1f}ms")
print(f"Avg enc_ms: {cuda_total_enc/n:.1f}, Overall RTFx: {cuda_total_audio*1000/cuda_total_inf:.0f}x")

print(f"\n=== Speed Ratio ===")
print(f"CUDA/TRT inference: {cuda_total_inf/trt_total_inf:.2f}x")
print(f"CUDA/TRT encoder:   {cuda_total_enc/trt_total_enc:.2f}x")
