#!/usr/bin/env python3
"""Detailed per-file TRT vs CUDA encoder comparison."""
import subprocess, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def get_timing(combined):
    m = re.search(r'([\d.]+)s audio, ([\d.]+)ms \(mel=([\d.]+) enc=([\d.]+) dec=([\d.]+)\), ([\d.]+)x RTFx', combined)
    if m:
        return float(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(5))
    return None, None, None, None

print(f"{'T':>5}  {'TRT_enc':>8}  {'CUDA_enc':>8}  {'ratio':>6}  {'TRT_dec':>8}  {'CUDA_dec':>8}  file")

results = []
for wav in sorted((ROOT / "data" / "librispeech").glob("*.wav")):
    trt = subprocess.run([str(ROOT / "parakeet"), str(wav)], capture_output=True, text=True)
    cuda = subprocess.run([str(ROOT / "parakeet_cuda"), str(wav)], capture_output=True, text=True)
    ta, tm, te, td = get_timing(trt.stdout + trt.stderr)
    ca, cm, ce, cd = get_timing(cuda.stdout + cuda.stderr)
    if te and ce:
        T = int(ta * 16000 / 160 / 8)
        results.append((T, te, ce, td, cd, wav.name))

results.sort()
total_trt_enc = total_cuda_enc = 0
for T, te, ce, td, cd, name in results:
    total_trt_enc += te; total_cuda_enc += ce
    ratio = ce / te if te > 0 else 0
    print(f"{T:5d}  {te:8.2f}  {ce:8.2f}  {ratio:6.2f}  {td:8.2f}  {cd:8.2f}  {name}")

n = len(results)
print(f"\n{'Avg':>5}  {total_trt_enc/n:8.2f}  {total_cuda_enc/n:8.2f}  {total_cuda_enc/total_trt_enc:6.2f}")
print(f"{'Total':>5}  {total_trt_enc:8.2f}  {total_cuda_enc:8.2f}")
