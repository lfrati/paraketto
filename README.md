<p align="center">
  <img src="paraketto.png" width="256" alt="parakettő">
</p>

# parakettő

Speech-to-text inference for NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), written in C++ with custom CUDA kernels. No frameworks, no Python at runtime.

- Batch 1, 1250x+ real-time — fast on a single WAV
- Custom CUDA/CUTLASS kernels — only `libcudart.so`
- Optional FP8 quantization — half the weight size, faster startup
- Optimized 1.8 GB VRAM usage
- ~240ms warm startup (FP16), ~180ms (FP8)
- Builtin HTTP server
- Optional static build with zero runtime files

```
WAV (16kHz/24kHz mono) → mel spectrogram → conformer encoder → TDT greedy decoder → text
```

## Performance

RTX 5070 Ti, batch size 1. Two FP16 GEMM backends: **CUTLASS** (zero dependencies beyond `libcudart.so`) and **cuBLAS** (requires `libcublas.so`). Plus an **FP8** backend using cublasLt E4M3 quantized weights. Everything else — FFT, mel filterbank, LayerNorm, convolutions, SiLU, GLU, LSTM, greedy decoding — runs on custom CUDA kernels in all backends.

```
                 CUTLASS (cudart only)          cuBLAS (+ libcublas)
              ────────────────────────────   ────────────────────────────
               RTFx    WER    Audio  Time     RTFx    WER    Audio  Time
librispeech   1069x   1.68%   896s  838ms    1077x   1.68%   896s  832ms
earnings22     979x  16.48%   253s  259ms    1000x  16.48%   253s  253ms
long          1307x   1.90%  5578s  4.27s    1306x   1.90%  5578s  4.27s
difficult     1205x  23.32%   509s  422ms    1261x  23.24%   509s  404ms
              ────────────────────────────   ────────────────────────────
Total         1250x          7236s  5.79s    1256x          7236s  5.76s
```

FP8 backend (requires Blackwell GPU):

```
                 FP8 (cublasLt E4M3)
              ────────────────────────────
               RTFx    WER    Audio  Time
librispeech   1152x   2.03%   896s  778ms
earnings22    1015x  15.76%   253s  249ms
long          1360x   2.20%  5578s  4.10s
difficult     1269x  19.38%   509s  401ms
              ────────────────────────────
Total         1309x          7236s  5.53s
```

### Startup time

Time from process start to first inference, measured with `tests/bench_startup.py`:

```
                startup (cold / warm)
CUTLASS:       600ms / 240ms      weights.bin (1.2 GB)
cuBLAS:        620ms / 240ms      weights.bin (1.2 GB)
FP8:           325ms / 180ms      weights_fp8.bin (604 MB)
```

Cold = weight files not in OS page cache. Warm = cached.

### Test machine

```
┌───────────┬────────────────────────────────────────────────────────────────┐
│ CPU       │ Intel Core i7-12700 — 2.1 GHz base / 4.9 GHz boost, 25 MB L3   │
│ RAM       │ Corsair Vengeance LPX 32 GB DDR4-3200 CL16, dual ch, 51.2 GB/s │
│ GPU       │ NVIDIA GeForce RTX 5070 Ti — 16 GB GDDR7, 896 GB/s, 2452 MHz   │
│ Storage   │ Samsung 970 EVO 1 TB NVMe — PCIe 3.0 x4, 3400/2500 MB/s r/w    │
└───────────┴────────────────────────────────────────────────────────────────┘
```

## Backends

Three CUDA backends, same driver and weight loader:

| Binary | GEMM backend | Weights | Notes |
|--------|-------------|---------|-------|
| `paraketto.cuda` | CUTLASS FP16 (custom-tuned) | `weights.bin` (1.2 GB) | default, no cuBLAS dep |
| `paraketto.cublas` | cuBLAS/cublasLt FP16 | `weights.bin` (1.2 GB) | |
| `paraketto.fp8` | cublasLt FP8 E4M3 | `weights_fp8.bin` (604 MB) | Blackwell only |

## Quick start

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer), CUDA toolkit 12+
- Python 3.10+ with [uv](https://docs.astral.sh/uv/) (for weight export and benchmarks only — not needed at runtime)

### Build & run

```bash
uv sync                          # install Python deps (weight export only)
make weights                     # download or export weights.bin (~1.2GB)
make paraketto.cuda              # CUTLASS backend (cudart only)
make paraketto.cublas            # cuBLAS backend
./paraketto.cuda audio.wav       # transcribe
```

### FP8 backend (Blackwell)

```bash
make paraketto.fp8               # build FP8 binary
./paraketto.fp8 audio.wav        # first run quantizes weights.bin → weights_fp8.bin
```

On first run without `weights_fp8.bin`, `paraketto.fp8` generates it from `weights.bin` and saves for future runs. Subsequent runs only load `weights_fp8.bin` (604 MB) — no `weights.bin` needed.

## Usage

```bash
./paraketto.cuda audio.wav               # single file
./paraketto.cuda *.wav                   # multiple files
./paraketto.cuda --weights FILE audio.wav  # custom weights path
```

### Server mode

```bash
./paraketto.cuda --server                    # listen on 0.0.0.0:8080
./paraketto.cuda --server :5001              # custom port
./paraketto.cuda --server 127.0.0.1:5001     # bind to localhost
```

All backends support the same server mode.

## HTTP API

- `GET /health` — returns `{"status":"ok"}`
- `POST /transcribe` — multipart `file` upload, returns `{"text":"...","audio_duration_s":...,"inference_time_s":...}`

```bash
curl localhost:8080/health
curl -F file=@audio.wav localhost:8080/transcribe
```

## Benchmarks

```bash
make bench-cuda    # WER + RTFx (CUTLASS backend)
make bench-cublas  # WER + RTFx (cuBLAS backend)
make bench-fp8     # WER + RTFx (FP8 backend)
make bench-all     # all backends

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ CUDA · paraketto_cuda.cpp + CUTLASS FP16

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.68% │   1069x │    100 │    896s │    838ms │
│ earnings22  │   16.48% │    979x │     40 │    253s │    259ms │
│ long        │    1.90% │   1307x │     50 │   5578s │    4.27s │
│ difficult   │   23.32% │   1205x │     50 │    509s │    422ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1250x │    240 │   7236s │    5.79s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ cuBLAS · paraketto_cuda.cpp + cuBLAS FP16

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    1.68% │   1077x │    100 │    896s │    832ms │
│ earnings22  │   16.48% │   1000x │     40 │    253s │    253ms │
│ long        │    1.90% │   1306x │     50 │   5578s │    4.27s │
│ difficult   │   23.24% │   1261x │     50 │    509s │    404ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1256x │    240 │   7236s │    5.76s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C++ FP8  · paraketto_cuda.cpp + cublasLt FP8

┌─────────────┬──────────┬─────────┬────────┬─────────┬──────────┐
│ Dataset     │      WER │    RTFx │   Utts │   Audio │     Time │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ librispeech │    2.03% │   1152x │    100 │    896s │    778ms │
│ earnings22  │   15.76% │   1015x │     40 │    253s │    249ms │
│ long        │    2.20% │   1360x │     50 │   5578s │    4.10s │
│ difficult   │   19.38% │   1269x │     50 │    509s │    401ms │
├─────────────┼──────────┼─────────┼────────┼─────────┼──────────┤
│ Total       │          │   1309x │    240 │   7236s │    5.53s │
└─────────────┴──────────┴─────────┴────────┴─────────┴──────────┘
```

## Static binary (no runtime files)

```bash
make paraketto.static      # embeds weights.bin, CUTLASS FP16
make paraketto.fp8.static  # embeds weights_fp8.bin, FP8
```

Requires only the NVIDIA driver + shared CUDA/cuBLAS libraries. No weights files at runtime.

## Project structure

```
src/paraketto_cuda.cpp    # CUDA backend main (mel, server, greedy decode)
src/conformer.cpp         # FP16 CudaModel (CUTLASS or cuBLAS via gemm.h)
src/conformer_fp8.cpp     # FP8 CudaModel (cublasLt E4M3, per-tensor scaling)
src/conformer_fp8.h       # FP8 CudaModel header (adds fp8_pool, scales, handles)
src/weights.cpp           # Weight loading (shared by all backends)
src/gemm.h                # Unified GEMM interface (backend selected at link time)
src/cutlass_gemm.cu       # CUTLASS FP16 backend
src/cublas_gemm.cu        # cuBLAS FP16 backend
src/kernels.cu            # Custom kernels: FFT, LayerNorm, SiLU, GLU, conv, LSTM, ...
src/kernels_fp8.cu        # FP8 kernels: absmax quantize, static quantize, transpose
src/mel.h                 # Custom 512-point FFT + mel filterbank
scripts/export_weights.py # NeMo → weights.bin converter
```

## References

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA's ASR model
- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
- [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA Templates for Linear Algebra Subroutines
