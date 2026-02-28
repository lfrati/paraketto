#!/usr/bin/env python3
"""Profile encoder sections by running parakeet_cuda with CUDA_LAUNCH_BLOCKING=1
and using CUDA events from a modified binary. Since we can't easily modify the binary,
we measure indirectly by comparing different scenarios."""
import subprocess, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Run profiling on the 60s file multiple times to get stable numbers
print("=== Encoder profiling: per-file breakdown ===")
print(f"{'T':>5}  {'enc_ms':>8}  {'dec_ms':>8}  {'mel_ms':>8}  {'total':>8}  {'RTFx':>6}  file")

results = []
for wav in sorted((ROOT / "data" / "librispeech").glob("*.wav")):
    out = subprocess.run([str(ROOT / "parakeet_cuda"), str(wav)],
                        capture_output=True, text=True)
    combined = out.stdout + out.stderr
    m = re.search(r'([\d.]+)s audio, ([\d.]+)ms \(mel=([\d.]+) enc=([\d.]+) dec=([\d.]+)\), ([\d.]+)x RTFx', combined)
    if m:
        audio_s = float(m.group(1))
        total_ms = float(m.group(2))
        mel_ms = float(m.group(3))
        enc_ms = float(m.group(4))
        dec_ms = float(m.group(5))
        rtfx = float(m.group(6))
        T = int(audio_s * 16000 / 160 / 8)
        results.append((T, enc_ms, dec_ms, mel_ms, total_ms, rtfx, wav.name))

results.sort()
total_enc = total_dec = total_mel = total_total = 0
for T, enc, dec, mel, total, rtfx, name in results:
    total_enc += enc; total_dec += dec; total_mel += mel; total_total += total
    print(f"{T:5d}  {enc:8.2f}  {dec:8.2f}  {mel:8.2f}  {total:8.2f}  {rtfx:6.0f}  {name}")

n = len(results)
print(f"\n{'Total':>5}  {total_enc:8.2f}  {total_dec:8.2f}  {total_mel:8.2f}  {total_total:8.2f}")
print(f"{'Mean':>5}  {total_enc/n:8.2f}  {total_dec/n:8.2f}  {total_mel/n:8.2f}  {total_total/n:8.2f}")

# Compute enc_ms / T correlation
print(f"\n=== Encoder time vs T ===")
print(f"Smallest T: {results[0][0]}, enc={results[0][1]:.2f}ms, ms/T={results[0][1]/results[0][0]*1000:.2f}us/T")
print(f"Largest T:  {results[-1][0]}, enc={results[-1][1]:.2f}ms, ms/T={results[-1][1]/results[-1][0]*1000:.2f}us/T")
# Quadratic component (attention is O(T^2))
# enc_ms ≈ a*T^2 + b*T + c
# Compare enc/T for small vs large to see if it's O(T^2)
if len(results) > 5:
    small = results[:5]
    large = results[-5:]
    small_avg_per_T = sum(r[1]/r[0] for r in small) / 5
    large_avg_per_T = sum(r[1]/r[0] for r in large) / 5
    print(f"Small 5 avg enc/T: {small_avg_per_T*1000:.2f} us/T")
    print(f"Large 5 avg enc/T: {large_avg_per_T*1000:.2f} us/T")
    print(f"Ratio: {large_avg_per_T/small_avg_per_T:.2f}x (>1 indicates superlinear scaling)")
