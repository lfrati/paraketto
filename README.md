# paraketto

Single-file C++ speech-to-text using NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) model.

```
WAV (16kHz mono) → mel spectrogram → FastConformer encoder → TDT greedy decoder → text
```

## Performance

RTX 5070 Ti (SM120 Blackwell), 240 utterances / 7236s audio:

```
                 RTFx    WER (libri)   startup
Python+TRT:      818x    1.68%         2.5s
C++ TRT:        1126x    1.68%         ~400ms
C++ CUTLASS:    1232x    1.68%         ~600ms
C++ cuBLAS:     1237x    1.68%         ~600ms
C++ FP8:        1238x    1.68%         631ms   (588MB weights_fp8.bin)
```

## Backends

Three CUDA backends, same driver and weight loader:

| Binary | GEMM backend | Weights needed | Notes |
|--------|-------------|----------------|-------|
| `paraketto.cuda` | CUTLASS FP16 (custom-tuned) | `weights.bin` (1.2GB) | default, no cuBLAS dep |
| `paraketto.cublas` | cuBLAS/cublasLt FP16 | `weights.bin` (1.2GB) | |
| `paraketto.fp8` | cublasLt FP8 E4M3 | `weights.bin` + `weights_fp8.bin` (588MB) | recommended |

## Quick start

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer recommended; Blackwell for FP8)
- CUDA toolkit 12+
- Python 3.10+ with [uv](https://docs.astral.sh/uv/) (for weight export and benchmarks only)

### Build & run

```bash
uv sync                          # install Python deps (weight export only)
make weights                     # download or export weights.bin (~1.2GB)
make paraketto.cuda              # CUTLASS FP16 backend
./paraketto.cuda audio.wav       # transcribe
```

### FP8 backend (recommended)

```bash
make paraketto.fp8               # build FP8 binary
make weights-fp8                 # quantize weights.bin → weights_fp8.bin (588MB, one-time)
./paraketto.fp8 audio.wav        # 631ms startup, same throughput as FP16
```

On first run without `weights_fp8.bin`, `paraketto.fp8` auto-generates it and saves for future runs.

### Server mode

```bash
./paraketto.fp8 --server              # listen on 0.0.0.0:8080
./paraketto.fp8 --server :5001
./paraketto.fp8 --server 127.0.0.1:5001
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
make bench-all   # all backends: Python, TRT, CUTLASS, cuBLAS, FP8
make bench-cuda  # CUTLASS FP16 only
```

## TensorRT backend (reference)

```bash
uv sync && make engines && make paraketto
./paraketto audio.wav
make bench-cpp
```

## Static binary (no runtime files)

```bash
make paraketto.static      # embeds weights.bin, CUTLASS FP16
make paraketto.fp8.static  # embeds weights.bin + weights_fp8.bin, FP8
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
src/kernels.cu            # Custom kernels: LayerNorm, SiLU, GLU, conv, LSTM, etc.
src/kernels_fp8.cu        # FP8 kernels: absmax quantize, static quantize, transpose
src/paraketto.cpp         # TensorRT backend (reference)
scripts/export_weights.py # NeMo → weights.bin converter
scripts/build_engines.py  # ONNX → TRT engine builder
```

## References

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA ASR model
- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
