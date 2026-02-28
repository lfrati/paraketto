#!/usr/bin/env python3
"""Benchmark parakeet_cuda on all WAV files and summarize results."""
import subprocess, sys, re, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "parakeet_cuda"
DATA_DIRS = [ROOT / "data" / "librispeech"]

wavs = []
for d in DATA_DIRS:
    wavs.extend(sorted(d.glob("*.wav")))

print(f"Benchmarking {len(wavs)} files...")

total_audio = 0.0
total_inf = 0.0
total_enc = 0.0
total_dec = 0.0
total_mel = 0.0
results = []

for wav in wavs:
    out = subprocess.run([str(BINARY), str(wav)], capture_output=True, text=True)
    combined = out.stdout + out.stderr
    # Parse: 11.1s audio, 17.0ms (mel=3.2 enc=9.1 dec=4.7), 652x RTFx
    m = re.search(r'([\d.]+)s audio, ([\d.]+)ms \(mel=([\d.]+) enc=([\d.]+) dec=([\d.]+)\), ([\d.]+)x RTFx', combined)
    if m:
        audio = float(m.group(1))
        inf = float(m.group(2))
        mel = float(m.group(3))
        enc = float(m.group(4))
        dec = float(m.group(5))
        rtfx = float(m.group(6))
        total_audio += audio
        total_inf += inf
        total_enc += enc
        total_dec += dec
        total_mel += mel
        results.append((wav.name, audio, inf, enc, dec, rtfx))
    else:
        print(f"  FAILED to parse: {wav.name}")
        print(f"  Output: {combined[:200]}")

n = len(results)
if n > 0:
    avg_enc = total_enc / n
    avg_dec = total_dec / n
    avg_mel = total_mel / n
    avg_inf = total_inf / n
    overall_rtfx = total_audio * 1000 / total_inf
    print(f"\n=== CUDA Backend Results ({n} files) ===")
    print(f"Total audio:     {total_audio:.1f}s")
    print(f"Total inference: {total_inf:.1f}ms")
    print(f"Avg per file:    {avg_inf:.1f}ms (mel={avg_mel:.1f} enc={avg_enc:.1f} dec={avg_dec:.1f})")
    print(f"Overall RTFx:    {overall_rtfx:.0f}x")
