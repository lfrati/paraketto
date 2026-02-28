# parakeet

Single-file C++ speech-to-text using NVIDIA's [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) model with TensorRT. ~950 lines, no Python at inference time.

```
WAV (16kHz mono) → CPU mel spectrogram (cuFFT) → TRT encoder → TRT decoder (greedy TDT) → text
```

## Performance

```
librispeech: WER=1.81%  RTFx=638x  (40 utts, 276s audio)
earnings22:  WER=16.48% RTFx=641x  (40 utts, 253s audio)
long-audio:  RTFx=873x  (92s audio, 105ms inference)
startup:     584ms (RTX 5070 Ti)
```

## Quickstart

Prerequisites: Linux, NVIDIA GPU, CUDA toolkit, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install Python deps (for engine building + benchmarks)
make engines                     # build TRT engines from ONNX models (~1.2GB encoder, 18MB decoder)
make parakeet                    # compile the C++ binary
./parakeet audio.wav             # transcribe a 16kHz mono WAV file
./parakeet --engine-dir DIR *.wav  # custom engine path, multiple files
```

## Server mode

Keep the pipeline loaded in memory and serve transcription over HTTP:

```bash
./parakeet --server              # listen on 0.0.0.0:8080
./parakeet --server :5001        # listen on 0.0.0.0:5001
./parakeet --server 127.0.0.1:5001  # bind to localhost only
```

Endpoints:
- `GET /health` — returns `{"status":"ok"}`
- `POST /transcribe` — multipart `file` upload, returns `{"text":"...","audio_duration_s":...,"inference_time_s":...}`

```bash
curl localhost:8080/health
curl -F file=@data/librispeech/6930-75918-0000.wav localhost:8080/transcribe
```

## Benchmarks

```bash
make bench       # run both Python and C++ benchmarks
make bench-cpp   # run C++ benchmark only
```

## Project structure

```
src/parakeet.cpp         # single-file C++ inference runtime
scripts/build_engines.py # ONNX → TensorRT engine builder
engines/
  encoder.engine         # 1.2 GB, FP16 (GPU-specific, not checked in)
  decoder_joint.engine   # 18 MB, FP16
```

## References

- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA batch ASR model
- [TDT paper](https://arxiv.org/abs/2304.06795) — Token-and-Duration Transducer (ICML 2023)
- [FastConformer paper](https://arxiv.org/abs/2305.05084) — encoder architecture
