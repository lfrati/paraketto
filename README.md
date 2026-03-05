# parakettő

Batch-1 speech-to-text inference for NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), written in C++ with custom CUDA kernels. No frameworks, no Python at runtime.

```
WAV (16kHz/24kHz mono) → mel spectrogram → conformer encoder → TDT greedy decoder → text
```

## Performance

RTX 5070 Ti, batch size 1. Two GEMM backends: **CUTLASS** (zero dependencies beyond `libcudart.so`) and **cuBLAS** (~2% faster, requires `libcublas.so`). Everything else — FFT, mel filterbank, LayerNorm, convolutions, SiLU, GLU, LSTM, greedy decoding — runs on custom CUDA kernels in both.

```
                 CUTLASS (cudart only)          cuBLAS (+ libcublas)
              ────────────────────────────   ────────────────────────────
               RTFx    WER    Audio  Time     RTFx    WER    Audio  Time
librispeech   1008x   1.68%   896s  888ms    1042x   1.68%   896s  860ms
earnings22     938x  16.48%   253s  270ms     965x  16.48%   253s  262ms
long          1263x   1.93%  5578s  4.42s    1277x   1.90%  5578s  4.37s
difficult     1178x  23.49%   509s  432ms    1226x  23.32%   509s  415ms
              ────────────────────────────   ────────────────────────────
Total         1204x          7236s  6.01s    1226x          7236s  5.90s
```

## Quick start

```bash
uv sync                          # install Python deps (weight export only)
make weights                     # export model weights to weights.bin (~1.2GB)
make paraketto.cuda              # CUTLASS backend (cudart only)
make paraketto.cublas            # cuBLAS backend (slightly faster)
./paraketto.cuda audio.wav       # transcribe
```

### Prerequisites

- Linux, NVIDIA GPU (Ampere or newer), CUDA toolkit 12+
- Python 3.10+, [uv](https://docs.astral.sh/uv/) (for weight export + benchmarks only — not needed at runtime)

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

### HTTP API

- `GET /health` — returns `{"status":"ok"}`
- `POST /transcribe` — multipart `file` upload, returns `{"text":"...","audio_duration_s":...,"inference_time_s":...}`

```bash
curl localhost:8080/health
curl -F file=@audio.wav localhost:8080/transcribe
```

### Benchmarks

```bash
make bench-cuda    # WER + RTFx (CUTLASS backend)
make bench-cublas  # WER + RTFx (cuBLAS backend)
```

## TensorRT backend (reference)

A reference TensorRT backend is also included. Requires TensorRT runtime libraries.

```bash
make engines                     # build TRT engines from ONNX
make paraketto                   # compile the TRT binary
./paraketto audio.wav
make bench-cpp                   # benchmark
```

## Project structure

```
src/paraketto_cuda.cpp   # main: WAV loading, mel, server, greedy decode loop
src/conformer.cpp        # conformer encoder + joint network
src/gemm.h               # unified GEMM interface (backend-agnostic)
src/cutlass_gemm.cu      # CUTLASS FP16 GEMM backend (no cuBLAS dep)
src/cublas_gemm.cu       # cuBLAS/cublasLt GEMM backend
src/kernels.cu           # custom CUDA kernels (FFT, LayerNorm, SiLU, GLU, conv, LSTM, ...)
src/mel.h                # custom 512-point FFT + mel filterbank
src/paraketto.cpp        # TensorRT backend (reference)
scripts/export_weights.py  # NeMo → weights.bin converter
```

## References

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA's ASR model
- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
- [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA Templates for Linear Algebra Subroutines
