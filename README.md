# parakeet

Local speech-to-text using NVIDIA's Parakeet TDT 0.6B V2 model via [onnx-asr](https://github.com/istupakov/onnx-asr), accelerated with TensorRT.

## Quickstart

Prerequisites: Linux, NVIDIA GPU, CUDA toolkit, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make bench   # WER + RTFx benchmark (80 utterances, ~9 min of audio)
make run     # live mic dictation (requires arecord)
```

## References

- [onnx-asr](https://github.com/istupakov/onnx-asr) — ONNX ASR library
- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA batch ASR model
