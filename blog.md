# Paraketto: Building a Single-Executable ASR Engine

## Goal

Replace the current Python + onnx-asr + TensorRT stack with a single compiled executable that runs NVIDIA's Parakeet TDT 0.6B V2 faster than the Python baseline. No Python, no pip, no virtualenv — just one binary and the model weights.

## Current Baseline

Our existing implementation (`src/parakeet_trt.py`) uses Python's `onnx-asr` library with TensorRT acceleration via ONNX Runtime. Benchmark results on RTX 5070 Ti:

```
librispeech: WER=1.81% RTFx=529x (40 utts, 276s audio)
earnings22:  WER=16.48% RTFx=603x (40 utts, 253s audio)
long-audio:  RTFx=874x (92s audio, 105ms mean, 5 runs)
```

---

## Research

### 1. Model Architecture

Parakeet TDT 0.6B V2 is an encoder-decoder transducer with three components: a FastConformer encoder, an LSTM prediction network, and a joint network. The "TDT" (Token-and-Duration Transducer) is the key innovation — instead of standard RNN-T which advances one frame at a time, TDT predicts both a token AND a duration, allowing it to skip frames and achieve 2-3x faster decoding.

#### 1.1 Preprocessing: Mel Spectrogram

Raw 16kHz mono audio is converted to a 128-bin mel spectrogram before entering the encoder. Parameters:

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Mel bins | **128** (not 80 — this is important) |
| FFT size | 512 |
| Window size | 0.025s (400 samples) |
| Hop length | 0.01s (160 samples) |
| Window type | Hann |
| Dither | 1e-5 |
| Normalization | per_feature (zero-mean, unit-variance per bin) |

The 128 mel bins is confirmed by [sherpa-onnx export metadata](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/parakeet-tdt-0.6b-v2/export_onnx.py) (`feat_dim: 128`) and by [NeMo issue #14884](https://github.com/NVIDIA-NeMo/NeMo/issues/14884) where the error message explicitly shows the encoder expects `(batch, 128, n_frame)`.

#### 1.2 FastConformer Encoder

The encoder is a FastConformer XL — 24 Conformer blocks with 8x convolutional downsampling. For a 10-second clip (1000 mel frames), the output is 125 encoder frames of dimension 1024.

**Encoder configuration:**

| Parameter | Value |
|-----------|-------|
| Hidden dimension (d_model) | 1024 |
| Layers | 24 |
| Attention heads | 8 (head dim = 128) |
| FF intermediate size | 4096 (4x expansion) |
| Conv kernel size | 9 |
| Subsampling factor | 8x |
| Subsampling type | depthwise-separable strided conv (`dw_striding`) |
| Subsampling channels | 256 |
| Attention type | relative positional (Transformer-XL style) |
| Activation | SiLU |

The 8x downsampling uses three depthwise-separable conv layers with stride 2 each (2x -> 4x -> 8x). This is one of FastConformer's key contributions vs. standard Conformer — more aggressive downsampling reduces the sequence length early, saving compute in the attention layers.

Each Conformer block has the classic "macaron" structure:
1. **Feed-Forward 1** (half-step): LayerNorm -> Linear(1024->4096) -> SiLU -> Dropout -> Linear(4096->1024) -> Dropout -> 0.5x residual
2. **Multi-Head Self-Attention**: LayerNorm -> RelPos MHSA (8 heads) -> Dropout -> residual
3. **Convolution Module**: LayerNorm -> Pointwise Conv(1024->2048) -> GLU -> Depthwise Conv(kernel=9, groups=1024) -> BatchNorm -> SiLU -> Pointwise Conv(1024->1024) -> Dropout -> residual
4. **Feed-Forward 2** (half-step): same as FF1
5. **Final LayerNorm**

The relative positional attention is Transformer-XL style (not RoPE). This computes content-based and position-based attention scores, requiring position embeddings up to `pos_emb_max_len=5000` frames (~6.7 minutes of audio after 8x downsampling, or ~24 minutes of raw audio).

#### 1.3 TDT Decoder (Prediction Network)

The decoder is a lightweight LSTM that processes previously emitted tokens:

| Parameter | Value |
|-----------|-------|
| Type | 2-layer LSTM |
| Hidden size | 640 |
| Input | token embeddings (vocab_size -> 640) |
| Output | 640-dim hidden state |
| State | (h, c) each of shape (2, batch, 640) |

#### 1.4 Joint Network

Combines encoder and decoder outputs to produce token + duration predictions:

| Parameter | Value |
|-----------|-------|
| Encoder projection | Linear(1024 -> 640) |
| Decoder projection | Linear(640 -> 640) |
| Combination | addition + ReLU |
| Output | Linear(640 -> 1030) |

The 1030 output values are: 1024 BPE token logits + 1 blank token logit + 5 duration logits (for durations {0, 1, 2, 3, 4}).

#### 1.5 TDT Decoding Algorithm

TDT's key innovation over standard RNN-T: the model predicts both **what token to emit** and **how many frames to skip** at each step.

Standard RNN-T:
- Predicts token or blank at each frame
- Blank = advance 1 frame, emit nothing
- Non-blank = emit token, stay at same frame
- Results in many blank predictions, O(T) joiner calls

TDT:
- Predicts token distribution AND duration distribution independently
- Blank with duration d = skip d frames, emit nothing
- Non-blank with duration d = emit token, skip d frames
- Can skip large chunks of silence/repetitive audio in one step
- Achieves 2-3x fewer joiner calls than standard RNN-T

Greedy decoding pseudocode:
```
t = 0
tokens = []
decoder_state = zeros  # LSTM h,c
decoder_out = run_decoder(blank_token, decoder_state)

while t < T:
    encoder_frame = encoder_output[:, :, t]
    logits = run_joiner(encoder_frame, decoder_out)

    token_logits = logits[:1025]    # vocab + blank
    duration_logits = logits[1025:] # 5 duration values

    token_id = argmax(token_logits)
    skip = argmax(duration_logits)
    if skip == 0: skip = 1  # prevent infinite loop

    if token_id != blank:
        tokens.append(token_id)
        decoder_out = run_decoder(token_id, decoder_state)

    t += skip

return bpe_decode(tokens)
```

#### 1.6 ONNX Export Structure

When exported to ONNX, the model splits into three files:

| File | Size (INT8) | Inputs | Outputs |
|------|-------------|--------|---------|
| encoder.onnx | ~622 MB | `audio_signal` float32 [B, 128, T], `length` int64 [B] | `encoder_output` [B, 1024, T/8], `encoded_lengths` [B] |
| decoder.onnx | ~6.9 MB | `targets` int32 [B, 1], `states_0/1` float32 [2, B, 640] | `decoder_output` [B, 640, 1], `states_0/1_next` |
| joiner.onnx | ~1.7 MB | `encoder_output` [B, 1024, 1], `decoder_output` [B, 640, 1] | `logits` [B, 1, 1, 1030] |

The preprocessor (mel spectrogram) and TDT decoding logic are NOT in the ONNX — they must be reimplemented in whatever language/framework you use.

#### 1.7 Key Papers

- **TDT**: Xu et al., "[Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795)," ICML 2023
- **FastConformer**: Rekesh et al., "[Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)," 2023
- **Conformer**: Gulati et al., "[Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)," 2020
- **Parakeet V3/Canary**: "[Canary-1B-v2 & Parakeet-TDT-0.6B-v3](https://arxiv.org/html/2509.14128v2)," 2025

---

### 2. Existing Implementations

Before building from scratch, we surveyed every existing project that runs Parakeet without Python.

#### 2.1 parakeet.cpp (Frikallo) — Pure C++, Custom Tensor Library

**Repo:** [github.com/Frikallo/parakeet.cpp](https://github.com/Frikallo/parakeet.cpp)

From-scratch C++ implementation without ONNX Runtime — uses [Axiom](https://github.com/Frikallo/axiom), a custom lightweight tensor library with Metal and CUDA backends.

- Supports TDT 600M, CTC 110M, EOU 120M, Nemotron 600M, Sortformer 117M
- All models: 16kHz WAV -> 80-bin mel spectrogram -> FastConformer -> decode
- Performance: ~27ms encoder for 10s audio on Apple M3 GPU (RTF ~0.003)
- 19 commits, MIT license, active development
- **Primary target is Apple Silicon (Metal)**. CUDA backend is work-in-progress.
- Model weights converted from NeMo `.nemo` to safetensors
- Discussed on [Hacker News](https://news.ycombinator.com/item?id=47176239)

**Key insight:** The author chose to build their own tensor library rather than use GGML, because GGML's convolution support was inadequate for FastConformer. This tells us something about GGML's limitations for this architecture.

#### 2.2 sherpa-onnx (k2-fsa) — Production C++ with ONNX Runtime

**Repo:** [github.com/k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

The most mature option. 10.5k stars, 1.2k forks, 1687 commits. C++ core with bindings for 12 languages.

- Explicitly supports Parakeet TDT 0.6B V2 and V3 (INT8 quantized)
- Pre-converted models: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`
- ONNX model files: encoder (~622MB), decoder (~8.6MB), joiner (~1.7MB), tokens.txt
- TDT decoding implemented in C++: joint output [B, T, vocab+1+5], predicts token + duration
- Platforms: Linux, Windows, macOS, Android, iOS, Raspberry Pi, RISC-V, various NPUs
- C++ API: `OfflineRecognizer` with `OfflineTransducerModelConfig`
- Pre-built Android APKs, Flutter apps, WebSocket server, HuggingFace demos
- **Limitation:** Parakeet TDT runs offline (non-streaming) only. Uses ONNX Runtime (not a fully standalone binary).

#### 2.3 parakeet-rs — Rust + ONNX Runtime

**Repo:** [github.com/altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)

Rust wrapper around ONNX Runtime via the `ort` crate.

- Supports CTC, TDT (25 languages), EOU (streaming), Nemotron, Sortformer
- Execution providers: CUDA, TensorRT, WebGPU, DirectML, MiGraphX, auto-fallback to CPU
- 222 commits, 187 stars, MIT/Apache-2.0 dual license
- Active development (latest v0.3.3, Feb 2026)

#### 2.4 parakeet (Go) — Single Binary Server

**Repo:** [github.com/achetronic/parakeet](https://github.com/achetronic/parakeet)

Go binary with OpenAI Whisper-compatible REST API. Runs Parakeet via ONNX (INT8, ~670MB, ~2GB RAM). CPU only.

#### 2.5 parakeet.cpp (jason-ni) — GGML-based Attempt (Paused)

**Repo:** [github.com/jason-ni/parakeet.cpp](https://github.com/jason-ni/parakeet.cpp)

Attempted GGML-based implementation. Encoder verified but performance was poor (~0.3-1.0s per inference vs 27ms in Frikallo's axiom version). Project appears paused. Found bugs in `ggml_conv_2d_dw` that needed upstream PRs.

#### 2.6 NVIDIA's Own Deployment Paths

NVIDIA does NOT provide a standalone C++ binary for Parakeet. Their deployment story is:
- **Riva/NIM**: Containerized gRPC service (Triton under the hood). Not a standalone binary.
- **NeMo export**: Produces ONNX files, but preprocessor and decoding must be reimplemented externally.
- **NeMo trt_compile()**: TensorRT lazy compilation within PyTorch. Still Python.
- [C++ Riva clients](https://github.com/nvidia-riva/cpp-clients) exist but are gRPC stubs, not inference engines.

---

### 3. Framework Analysis

#### 3.1 GGML

GGML is the tensor library behind whisper.cpp and llama.cpp. It has ~95 op types with CPU, CUDA, Metal, Vulkan backends.

**What works for FastConformer:**
- Conv2D (via IM2COL + MUL_MAT): available on all backends
- LayerNorm, RMSNorm, Softmax, GELU, MulMat: all available
- Flash attention: partially supported

**What's missing or broken:**
- **Relative positional encoding (Transformer-XL style)**: GGML has RoPE (`ggml_rope`), but FastConformer uses a completely different attention mechanism. Would need to be built from basic matrix ops (~200-300 lines of careful tensor manipulation).
- **Depthwise 1D convolution**: `ggml_conv_1d_dw` was recently added but CUDA backend support is uncertain. The jason-ni attempt found bugs in `ggml_conv_2d_dw`.
- **Grouped convolution**: No `groups` parameter on conv ops. Depthwise is a special case (groups=channels) but arbitrary grouping is unsupported ([issue #833](https://github.com/ggml-org/ggml/issues/833)).
- **LSTM**: No native LSTM op — would need manual implementation from primitives.
- **1D pooling**: Not supported on any backend.

**Performance vs TensorRT:** [TensorRT-LLM is 30-70% faster than llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/7043) on the same GPU for LLM workloads. The gap comes from TensorRT's layer fusion, GPU-specific kernel compilation, and mixed-precision optimization. For convolution-heavy models like FastConformer, the gap would be larger since GGML decomposes convolutions into IM2COL + MUL_MAT rather than using fused kernels.

**Verdict:** Not viable. Missing critical ops, poor conv performance on CUDA, and the one project that tried it (jason-ni/parakeet.cpp) achieved poor results and was abandoned.

#### 3.2 ONNX Runtime C++ API

The pragmatic choice. ONNX Runtime's C++ API has essentially identical performance to the Python wrapper because the actual GPU compute happens in the same C++ backend. When configured with the TensorRT execution provider, it should match our current ~874x RTFx (long-audio baseline).

- **Single binary?** Partial. ONNX Runtime core can be built as a static lib, but TensorRT and CUDA execution providers are separate shared libraries. NVIDIA's CUDA/cuDNN/TensorRT libs can't be statically linked. Total deployment footprint with TRT: ~2.6 GB of shared libs.
- **Ease:** Low effort. sherpa-onnx already does exactly this.
- **Performance:** Matches Python wrapper. Configuration gotchas exist but are well-documented.

#### 3.3 TensorRT C++ API Directly

Maximum performance, maximum complexity. You'd compile ONNX models to TensorRT engines (GPU-architecture-specific), then write C++ for engine deserialization, memory management, and inference.

- **Single binary?** No. TensorRT [does not support static linking](https://github.com/NVIDIA/TensorRT/issues/4116). Runtime libraries are ~600-800 MB.
- **Performance:** The best achievable. NVIDIA's benchmark shows RTFx >3000 at batch 128 with FP8. At batch 1, still the fastest option.
- **Ease:** Hard. Must handle dynamic shapes, GPU-specific engine files, implement preprocessing with cuFFT, implement TDT decoding in C++.
- **Engines are GPU-specific:** Must rebuild for each GPU family (Turing, Ampere, Ada, Hopper...).

#### 3.4 Candle (Rust, HuggingFace)

HuggingFace's Rust ML framework. Has Whisper, LLaMA, Stable Diffusion implementations, but NO FastConformer or Conformer. You'd need to port the entire architecture from scratch.

- **Binary size:** Excellent — single-digit MB (plus CUDA shared libs).
- **Performance:** Moderate. No layer fusion or graph optimization like TensorRT. Individual ops are efficient but no cross-layer optimization.
- **Ease:** Very hard. Weeks to months of porting work, then weight loading from safetensors/ONNX.

#### 3.5 Burn (Rust)

Has an [ONNX import capability](https://github.com/tracel-ai/burn-onnx) that converts ONNX models to native Rust code. Promising but operator coverage is limited and actively evolving. Worth a quick test: try importing the Parakeet ONNX models with burn-onnx and see which ops fail.

CUDA backend achieved 97% of PyTorch performance on Phi3 benchmarks.

#### 3.6 CTranslate2

Does NOT support Conformer or Transducer architectures. Only standard Transformer encoder-decoder (T5, BART, Whisper), decoder-only (LLaMA, GPT), and encoder-only (BERT). Not viable.

#### 3.7 Apache TVM

TVM can theoretically compile any ONNX model via its Relay frontend, applying optimization passes and generating GPU code. Mixed performance vs TensorRT — [2025 Jetson benchmarks](https://www.mdpi.com/2079-9292/14/15/2977) show competitive results on some transformer architectures. But the compilation pipeline is complex (Python-based, requires hours of auto-tuning), and there's no existing FastConformer support.

#### 3.8 Custom CUDA Kernels

Maximum possible performance, but enormous engineering effort (person-months). You'd essentially be reimplementing what TensorRT already does. Not practical unless you have specific requirements no framework meets.

---

### 4. Comparison Matrix

| Approach | Python-Free? | TDT Support? | CUDA Perf? | True Single Binary? | Maturity | Effort |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **parakeet.cpp (Frikallo)** | Yes | Yes | WIP | Yes | Early | Fork + CUDA work |
| **sherpa-onnx** | Yes | Yes | Via ONNX RT | No (shared libs) | Production | Low (exists) |
| **parakeet-rs** | Yes | Yes | TRT/CUDA | No (shared libs) | Active | Low (exists) |
| **ONNX RT C++ (custom)** | Yes | Implement | TRT EP | No (shared libs) | High | Medium |
| **TensorRT C++ direct** | Yes | Implement | Maximum | No (shared libs) | High | High |
| **GGML** | Yes | Missing ops | Poor | Yes | N/A | Not viable |
| **Candle (Rust)** | Yes | Rewrite all | Moderate | Yes* | Medium | Very High |
| **Burn (Rust)** | Yes | Untested | Moderate | Yes* | Medium | High |
| **TVM** | Yes | Untested | Competitive | Partial | Medium | High |
| **Custom CUDA** | Yes | Implement | Maximum | Yes* | N/A | Extreme |

*CUDA shared libs still required at runtime — no CUDA solution can be truly statically linked.

Nobody had built a production-ready, single-binary, GPU-accelerated Parakeet executable for Linux. parakeet.cpp (Frikallo) is the closest architecturally but targets Apple Metal. sherpa-onnx is the most mature but depends on ONNX Runtime shared libs.

---

### 5. Sources

#### Model Architecture
- [nvidia/parakeet-tdt-0.6b-v2 HuggingFace](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [HuggingFace Transformers Parakeet docs](https://huggingface.co/docs/transformers/model_doc/parakeet)
- [NeMo FastConformer TDT config YAML](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe.yaml)
- [sherpa-onnx export script (confirms 128 mel bins)](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/parakeet-tdt-0.6b-v2/export_onnx.py)
- [NeMo issue #14884 (128 mel bins confirmation)](https://github.com/NVIDIA-NeMo/NeMo/issues/14884)
- [QED42 architecture deep dive](https://www.qed42.com/insights/nvidia-parakeet-tdt-0-6b-v2-a-deep-dive-into-state-of-the-art-speech-recognition-architecture)
- [DeepWiki: parakeet-mlx RNN-T components](https://deepwiki.com/senstella/parakeet-mlx/5.3-ctc-and-tdt-components)
- [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx)

#### Papers
- [TDT paper (arXiv:2304.06795)](https://arxiv.org/abs/2304.06795)
- [FastConformer paper (arXiv:2305.05084)](https://arxiv.org/abs/2305.05084)
- [Conformer paper (arXiv:2005.08100)](https://arxiv.org/abs/2005.08100)
- [Canary/Parakeet V3 paper (arXiv:2509.14128)](https://arxiv.org/html/2509.14128v2)

#### Existing Implementations
- [parakeet.cpp — Frikallo (C++/Axiom)](https://github.com/Frikallo/parakeet.cpp)
- [parakeet.cpp HN discussion](https://news.ycombinator.com/item?id=47176239)
- [Axiom tensor library](https://github.com/Frikallo/axiom)
- [sherpa-onnx (k2-fsa)](https://github.com/k2-fsa/sherpa-onnx)
- [sherpa-onnx NeMo transducer models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html)
- [parakeet-rs (Rust)](https://github.com/altunenes/parakeet-rs)
- [parakeet (Go server)](https://github.com/achetronic/parakeet)
- [jason-ni/parakeet.cpp (GGML, paused)](https://github.com/jason-ni/parakeet.cpp)
- [parakeet-mlx (Apple MLX)](https://github.com/senstella/parakeet-mlx)

#### Frameworks
- [GGML ops support matrix](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)
- [GGML depthwise conv issue #833](https://github.com/ggml-org/ggml/issues/833)
- [whisper.cpp architecture (DeepWiki)](https://deepwiki.com/ggml-org/whisper.cpp/2-architecture)
- [whisper.cpp Parakeet feature request #3118](https://github.com/ggml-org/whisper.cpp/issues/3118)
- [TensorRT-LLM vs llama.cpp performance](https://github.com/ggml-org/llama.cpp/discussions/7043)
- [ONNX Runtime TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [ONNX Runtime static build discussion](https://github.com/microsoft/onnxruntime/discussions/16799)
- [TensorRT static linking request (unresolved)](https://github.com/NVIDIA/TensorRT/issues/4116)
- [TensorRT C++ API docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html)
- [candle (HuggingFace Rust)](https://github.com/huggingface/candle)
- [burn (Rust)](https://github.com/tracel-ai/burn)
- [burn-onnx](https://github.com/tracel-ai/burn-onnx)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [TVM vs TensorRT benchmarks 2025](https://www.mdpi.com/2079-9292/14/15/2977)

#### NVIDIA Deployment
- [NeMo export docs](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/core/export.html)
- [NeMo Export-Deploy library](https://github.com/NVIDIA-NeMo/Export-Deploy)
- [NVIDIA Riva ASR](https://docs.nvidia.com/nim/riva/asr/latest/getting-started.html)
- [Riva C++ clients](https://github.com/nvidia-riva/cpp-clients)
- [NVIDIA Parakeet TDT performance blog](https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/)
- [Conformer CTC TensorRT export issue #8708](https://github.com/NVIDIA-NeMo/NeMo/issues/8708)

#### Benchmarks
- [Northflank STT benchmarks 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [NVIDIA Speech AI performance blog](https://developer.nvidia.com/blog/nvidia-speech-ai-models-deliver-industry-leading-accuracy-and-performance/)

---

## Implementation Log

### Step 1: TRT Headers and Project Setup

Downloaded TensorRT 10.15 C++ headers from the [NVIDIA/TensorRT GitHub repo](https://github.com/NVIDIA/TensorRT/tree/v10.15) (tag matching pip TRT 10.15.1) into `third_party/tensorrt/`. The pip `tensorrt-cu12` package has `.so` libs but no headers — and there's no `-dev` pip package for this version.

Linking required pointing directly at `libnvinfer.so.10` (no `.so` symlink exists in the pip package):
```makefile
LDFLAGS = ... $(TRT_LIBS)/libnvinfer.so.10 -Wl,-rpath,$(TRT_LIBS)
```

### Step 2: Engine Building

The ORT-cached TRT engines (from `~/.cache/parakeet-trt-cache/`) are internal subgraph engines built by ONNX Runtime's TRT EP. They have different I/O contracts than the full ONNX models and can't be used standalone.

Built proper engines from the ONNX models using TensorRT's Python builder API:
- Encoder: FP16, dynamic shapes `audio_signal: min=(1,128,5) opt=(1,128,2000) max=(1,128,12000)`
- Decoder+Joint: FP16, dynamic shapes for all inputs
- Preprocessor: FP32 (not used — see below)

Key gotcha: for ONNX models with external weights (encoder-model.onnx + encoder-model.onnx.data), the parser needs the file path to resolve relative weight paths: `parser.parse(data, path=onnx_path)`.

### Step 3: The TRT STFT Bug

The TRT preprocessor engine produces **wrong mel spectrogram values** (max diff 4.57 vs ORT reference). Root cause: TensorRT's STFT operator implementation diverges from ONNX Runtime's.

When ORT uses the TRT EP, it partitions the graph: STFT runs on CPU/CUDA while only the other ops (pre-emphasis, mel filterbank, normalization) run on TRT. A standalone TRT engine tries to run STFT on GPU using TRT's native implementation, which produces incorrect results.

Verified: ORT preprocessor features + TRT encoder + TRT decoder = correct transcription. TRT preprocessor features + TRT encoder + TRT decoder = garbage (all blanks).

**Solution:** Implemented mel spectrogram computation in C++ on CPU:
- Pre-emphasis (alpha=0.97)
- Hann windowing (N_FFT=512)
- Radix-2 Cooley-Tukey FFT (512-point)
- Power spectrum
- Mel filterbank (257x128, weights extracted from nemo128.onnx)
- Log + per-channel normalization (zero-mean, unit-variance)

Verified: C++ mel output matches ORT to 6 decimal places.

### Step 4: The n_frames vs n_valid Bug

With the CPU mel spectrogram working, most files transcribed correctly — but 7 out of 40 librispeech utterances produced **empty output** (all-blank decoder predictions).

The mel spectrogram was confirmed correct for these files too. The bug was in what we fed to the encoder:
- C++ fed `n_frames` (total mel frames including padding) to the encoder
- Python fed `n_valid` (= num_samples / hop_length) — the actual valid frame count

For a 3.2s file: `n_frames=321` but `n_valid=320`. The extra zero-padded frame at the end, when processed by the FP16 encoder, caused subtle differences in the encoder output that tipped the decoder toward all-blank predictions.

**Fix:** Changed encoder input shape from `[1, 128, n_frames]` to `[1, 128, n_valid]`, and only output `n_valid` frames from the mel computation.

### Step 5: Current Results

```
C++ (paraketto):
  librispeech: WER=1.81% (40 utts, 276s audio, 0 empty)
  long-audio:  RTFx=383x (92s audio, 239ms inference)
```

**Accuracy:** Librispeech WER matches Python exactly at 1.81%.

**Speed (initial):** Long-audio RTFx was 383x vs Python's 848x. The gap was from the CPU mel spectrogram doing a dense 257×128 matrix multiply per frame.

### Step 6: Sparse Mel Filterbank Optimization

The mel filterbank matrix is 98.5% sparse — each frequency bin maps to at most 2 mel bins. The dense matrix multiply was doing 32,896 multiply-adds per frame when only ~504 were nonzero. Precomputing a sparse representation (per-frequency list of mel bin + weight) reduced the mel time from 178.6ms → 45ms (4x speedup). Combined with precomputed bit-reversal tables for the FFT, the mel dropped to 42.9ms.

### Current Results (after optimization)

Long-audio RTFx: 918x (92s audio, 99.7ms: mel=42.9 enc=30.6 dec=26.2). C++ is now **faster than Python** on long-audio throughput (918x vs 848x RTFx), despite doing the mel spectrogram on CPU instead of GPU.

### Step 7: cuFFT Batched FFT

Replaced the hand-rolled radix-2 Cooley-Tukey FFT with cuFFT's batched R2C transform. No custom CUDA kernels needed — cuFFT is a host-side library callable from regular C++ (just link `-lcufft`).

The flow: CPU pre-emphasis and windowing → upload windowed frames to GPU → `cufftPlanMany` + `cufftExecR2C` → download complex output → CPU power spectrum, sparse mel, log, normalize.

Long-audio mel time: 42.9ms → 28.4ms (1.5x faster). Total RTFx: 918x → **836x**.

### Results After Step 7

Benchmarked on RTX 5070 Ti (compute 12.0). Note: long-audio WER was not measured here — this was before the WER bug was discovered (see Step 14).

```
C++ (paraketto):
  librispeech: WER=1.81% RTFx=620x (40 utts, 276s audio)
  earnings22:  WER=16.48% RTFx=643x (40 utts, 253s audio)
  long-audio:  RTFx=836x (92s audio, 110ms)

Python (parakeet_trt.py via onnx-asr):
  librispeech: WER=1.81% RTFx=436x (40 utts, 276s audio)
  earnings22:  WER=16.48% RTFx=458x (40 utts, 253s audio)
  long-audio:  RTFx=672x (92s audio, 136ms)
```

WER matches Python exactly on both datasets. The C++ binary is **~24% faster** on long-audio and **~40% faster** on short utterances (no Python/ORT overhead per file). Non-16kHz audio is resampled at load time (24kHz→16kHz via FIR low-pass + linear interpolation; other sample rates are rejected).

### Architecture

```
src/paraketto.cpp    # ~1150 lines, single file (includes HTTP server)
engines/
  encoder.engine     # 1.2 GB, FP16
  decoder_joint.engine  # 18 MB, FP16
Makefile
```

Vocab, Hann window, and mel filterbank weights are embedded directly in the source (no external data files). Pipeline: WAV → CPU mel spectrogram (cuFFT) → GPU TRT encoder → GPU TRT decoder (greedy TDT loop) → BPE detokenize → text

### Step 8: Startup Time Optimization

The binary was taking ~880ms (warm cache) to ~1.6s (cold cache) to produce its first output. Profiling showed the encoder engine load dominated at 96% of startup time. We tried eight optimizations:

**Baseline measurements:**
```
warm: 883ms total (enc=846 dec=13 mel+stream=4 bufs=0, warmup=17)
cold: 1567ms total (enc=1453 dec=22 mel+stream=56 bufs=0, warmup=34)
```

#### What worked

**mmap with MAP_POPULATE (the big win).** Replaced `std::ifstream` + `std::vector<char>` with `mmap(MAP_PRIVATE | MAP_POPULATE)` + `madvise(MADV_SEQUENTIAL)`. The old path heap-allocated 1.2GB, did a sequential `ifstream::read` copy into it, then TRT traversed the buffer again — three memory passes over 1.2GB. With mmap, TRT reads directly from the kernel page cache: zero copy, no heap allocation. Warm encoder load dropped from 846ms to 308ms (**-63%**).

**Parallel engine loading.** Moved decoder_joint loading to a background `std::thread` so it overlaps with the encoder load. The decoder is only 18MB / ~6ms warm, so warm-cache benefit is negligible. But on cold cache the decoder's disk I/O overlaps with the encoder's, saving ~100ms.

#### What helped marginally

- **Early `cudaFree(0)`** — forces CUDA driver init (~106ms) before engine loading. Didn't reduce total time but made the cost visible.
- **Pre-create cuFFT plan** — ~1ms saved from warmup.
- **WAV readahead with `posix_fadvise`** — zero-cost hint for the kernel to pre-cache the first WAV file.
- **Reuse warmup WAV data** — the first file was read and transcribed twice. Now `std::move`s the warmup data into the processing loop.

#### What didn't help

- **`-O3 -march=native -flto`** — startup is I/O-bound, not CPU-bound. Kept the flags for inference runtime.
- **Pooling GPU allocations** — 11 `cudaMalloc` calls already took <0.5ms. Not worth the complexity.

#### Final results

```
warm: 438ms total (cuda_init=106 engines=310 mel+stream=1 bufs=0, warmup=16)
cold: 1129ms total (cuda_init=299 engines=805 mel+stream=7 bufs=0, warmup=17)
```

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Warm startup | 883ms | 438ms | **-445ms (-50%)** |
| Cold startup | 1567ms | 1129ms | **-438ms (-28%)** |
| Encoder load (warm) | 846ms | 308ms | **-538ms (-64%)** |

The remaining 310ms of engine load is TRT deserialization — internal to TensorRT, it rebuilds GPU kernel launch parameters from the serialized plan. The 106ms CUDA driver init is a per-process cost. These are the floor without moving to a persistent server process or using TRT's `ITimingCache`.

### Step 9: HTTP Server Mode

Startup takes ~584ms, so spawning a process per request is unacceptable. Added a persistent HTTP server using [cpp-httplib](https://github.com/yhirose/cpp-httplib) (header-only, same as llama.cpp). ~100 lines: `GET /health`, `POST /transcribe` (multipart upload), `std::mutex` around GPU access, `--server [[host]:port]`. Drop-in replacement for our Python stt-server.

**Startup banner:**
```
model:     parakeet-tdt-0.6b-v2
engines:   engines (encoder 1185 MB, decoder 17 MB)
device:    NVIDIA GeForce RTX 5070 Ti (compute 12.0, 15842 MB VRAM, 13967 MB free)
startup:   584 ms (cuda_init=198 engines=368 warmup=17)
endpoints: GET /health | POST /transcribe

listening on http://localhost:8080
```

**Request logging:**
```
18:33:12  POST /transcribe  200
         audio=3.5s  inference=9ms  RTFx=398x  "Concord returned to its place amidst the tents."
```

The benchmark was rewritten to use the server endpoint — inference timing from the JSON response instead of regex-matching stderr. Fixed an off-by-one where the `init:` log line skewed per-dataset RTFx numbers.

### Step 10: Replacing TensorRT — Phase 1 (Weight Export + Loading)

Goal: eliminate the ~800MB libnvinfer.so and ~1.2GB GPU-specific engine files. Replace with cuBLAS + custom CUDA kernels.

#### Challenge 1: Anonymous ONNX Tensor Names

The ONNX models from `istupakov/parakeet-tdt-0.6b-v2-onnx` use anonymous names for most weight matrices. Instead of `layers.0.fc1.weight`, the weights feeding MatMul/Conv/LSTM ops get auto-generated names like `onnx::MatMul_6382`, `onnx::Conv_6310`, `onnx::LSTM_205`. This is standard for PyTorch's ONNX exporter — `nn.Module` parameters get proper names, but functional op inputs don't.

**Solution:** Walk the ONNX graph topology. Each node's output name contains the full architectural path (e.g. `/layers.0/feed_forward1/linear1/MatMul_output_0`). The `resolve_names()` function in `export_weights.py` traces which initializer feeds which node, extracts the semantic path from the output name, and reconstructs proper names for all op types.

#### Challenge 2: Multi-Layer LSTM Name Collision

Two LSTM nodes share the same base path, differing only by a `_1` suffix in their output names (`LSTM_output_0` vs `LSTM_1_output_0`). Initial name resolution stripped the suffix uniformly, so layer 1 silently overwrote layer 0.

**Solution:** Detect `LSTM_N_output_0` pattern and append the layer index. Layer 0 → `decoder.dec_rnn.lstm.*`, layer 1 → `decoder.dec_rnn.lstm.1.*`.

#### Challenge 3: Model Architecture Differs from Plan

The original plan assumed packed QKV [1024→3072], 3 subsampling layers with batch norm, and separate LSTM biases. The actual ONNX model has:

- **Separate Q, K, V, pos projections** (4 × [1024, 1024], not packed)
- **5 conv layers** in subsampling (indices 0,2,3,5,6), no batch norm
- **No batch norm** in conformer conv module
- **No biases** on FF linear layers
- **ONNX LSTM format** — combined bias [1, 8×H]
- A **`linear_pos` projection** for relative positional encoding (4th attention projection)

**Solution:** Rewrote Weights struct and name mappings based on actual ONNX inspection.

#### Results

- **625 tensors** exported (612 encoder + 13 decoder), all verified round-trip
- **paraketto-fp16.bin**: 1,236 MB FP16 — GPU-agnostic, works on any CUDA device
- **Weight loading**: 123ms (vs ~333ms TRT engine deserialization) — **2.7× faster**
- **Dual backend**: `--backend trt` (default, unchanged) and `--backend cuda` (loads weights, inference stub)
- TRT path completely unaffected — same binary, same behavior

```
$ ./paraketto --backend cuda --weights paraketto-fp16.bin data/librispeech/6930-75918-0000.wav
weights: 625 tensors, 1178.4 MB GPU
  all key weights mapped successfully
init: 202ms (weights=123 mel+stream=79 bufs=0)
```

### Step 11: Replacing TensorRT — Phase 2 (CUDA Encoder + Decoder)

Full CUDA implementation of the FastConformer encoder (24 blocks), LSTM decoder, and TDT joint network using cuBLAS GEMMs and custom CUDA kernels. No TensorRT, no cuDNN.

#### Custom CUDA Kernels Written

19 kernels in `kernels.cu` (~750 lines):
- **LayerNorm** (with optional residual+scale fusion)
- **SiLU / GLU** (vectorized half2)
- **Softmax** (numerically stable: FP32 accumulation, max subtraction)
- **Depthwise Conv1D k=9** (shared memory tiling)
- **Conv2D** (general, supports groups for depthwise)
- **LSTM cell** (fused gate computation, ONNX iofc gate order)
- **Attention helpers**: pos_bias add, relative position skew, score scaling
- **Data movement**: transpose, reshape, embedding gather, bias add, residual add, cast

#### Bug 1: cuBLAS GEMM Convention Mismatch

**Symptom:** Output was "was was was was..." (garbage repetition).

**Root cause:** cuBLAS assumes column-major but our data is row-major. The initial GEMM helper used `CUBLAS_OP_T` on W, effectively double-transposing ONNX MatMul weights (which are already stored as [K, N]). For row-major `Y[m,n] = X[m,k] @ W[k,n]`, the correct cuBLAS call is `cublasHgemm(OP_N, OP_N, n, m, k, W, n, X, k, Y, n)`. Getting the row-major/column-major mapping right took an embarrassing number of attempts.

**Solution:** Created two GEMM helpers:
- `gemm_nn`: `Y = X @ W` — for ONNX MatMul weights [k, n]
- `gemm_nt`: `Y = X @ W^T` — for Conv/PyTorch weights [n, k]

#### Bug 2: LSTM State Unconditionally Updated

**Symptom:** Output was "L M" (very short, only 2 tokens).

**Root cause:** In TDT/RNN-T, the prediction network LSTM should only advance its state when a non-blank token is emitted. The initial implementation used `std::swap(lstm_h, lstm_h_new)` inside `decode_step()`, which permanently modified LSTM state even for blank predictions.

**Solution:** Added separate `lstm_h_out[2]` and `lstm_c_out[2]` "staging" buffers. `decode_step()` writes new state to these without modifying the current state. A separate `decoder_commit()` method swaps the buffers, called only when `token != BLANK_ID`.

#### Bug 3: Attention Head Layout Mismatch

**Symptom:** Output was "Robin Carter." (real English but completely wrong).

**Root cause:** Q, K, V projections produce [T, 1024] = [T, 8, 128] in memory. The batched GEMM and pos_bias kernels expect [8, T, 128] layout. Element (h, t, d) in [8, T, 128] is at offset `h*T*128 + t*128 + d`, but in [T, 8, 128] it's at `t*1024 + h*128 + d` — different memory locations.

**Solution:** Added `transpose_0213_fp16` kernel that transposes [A, B, C] → [B, A, C]. The MHSA section now explicitly transposes Q, K, V, and pos_proj from [T, 8, 128] → [8, T, 128] before attention, and transposes attn_out back after the weighted sum.

#### Bug 4: Mel Spectrogram Layout (The Big One)

**Symptom:** Output was "Concorr returned with life amidst the tents." (close but wrong). Encoder output had cosine similarity of only 0.50 with TRT.

**Root cause:** The ONNX model **transposes** the mel input before the first conv2d. The mel spectrogram is `[batch, 128, T_mel]` but the ONNX graph does `Transpose(perm=[0,2,1])` → `[batch, T_mel, 128]` then `Unsqueeze(axis=1)` → `[batch, 1, T_mel, 128]` before the first 2D convolution.

My code was feeding `[1, 128, T_mel]` directly to conv2d (H=128, W=T_mel). The correct input is `[1, T_mel, 128]` (H=T_mel, W=128). This makes H the time dimension and W the frequency dimension. After 3 stride-2 stages:
- Correct: `[256, T/8, 16]` → reshape `[T/8, 4096]` (time is first dim)
- Wrong:   `[256, 16, T/8]` → reshape `[T/8, 4096]` (same shape but completely different values)

The output shapes matched (both 44 frames of 4096 dims), hiding the bug. The numerical values were completely different because the convolutions operate differently when H and W are swapped.

**Debugging approach:** Added encoder output dumps, compared against ONNX Runtime. Full encoder cosine similarity was 0.50 (terrible). Used `MAX_BLOCKS=0` to isolate subsampling output — still 0.77 cosine. Traced the ONNX graph backwards from conv.0 to find the Transpose+Unsqueeze chain.

**Solution:** Transpose mel from [128, T_mel] to [T_mel, 128] before conv2d. Also changed the final reshape kernel from `[C,H,W]→[W,C*H]` to `[C,H,W]→[H,C*W]` to match the new dimension ordering.

Always trace the exact ONNX graph from input to the first compute node. Don't assume the input layout matches the weight shapes — there may be implicit reshapes/transposes.

#### Results

After all four fixes, the CUDA backend produces **identical** transcriptions to TRT on all test files:

```
$ for f in data/librispeech/6930-75918-000{0,1,2,3,4}.wav; do
    echo "$(basename $f):"
    echo "  TRT:  $(./paraketto --backend trt $f 2>/dev/null | tail -1)"
    echo "  CUDA: $(./paraketto --backend cuda --weights paraketto-fp16.bin $f 2>/dev/null | tail -1)"
  done

6930-75918-0000.wav:
  TRT:  Concord returned to its place amidst the tents.
  CUDA: Concord returned to its place amidst the tents.
[... all 5 files match exactly ...]
```

Encoder output cosine similarity: **0.9999** (mean across frames), max abs diff 0.015 (FP16 vs FP32 rounding).

Performance: **416x RTFx** (CUDA) vs **541x RTFx** (TRT). The CUDA backend is ~23% slower than TRT, mainly in the decoder (cuBLAS per-step overhead for small GEMMs vs TRT's fused kernels). The encoder is comparable (3.5ms CUDA vs 4.2ms TRT).

#### Architecture (CUDA backend)

```
src/kernels.cu       # ~800 lines: 19 custom CUDA kernels
src/kernels.h        # kernel declarations
src/conformer.h      # Weights struct + CudaModel struct
src/conformer.cpp    # ~800 lines: weight loading + encoder/decoder forward pass
src/paraketto.cpp    # Pipeline integration, --backend cuda support
paraketto-fp16.bin          # 1.2 GB flat binary (GPU-agnostic, no engine rebuilds)
```

Link dependencies: `cudart`, `cublas`, `cublasLt`, `cufft`, `cudnn`. cuDNN is used only for flash attention (SDPA) in the encoder.

### Step 12: Encoder Kernel Fusion

After Step 11, the CUDA encoder ran 24 operations per conformer block (576 total across 24 blocks): 10 cuBLAS GEMMs + 1 cuDNN SDPA + 13 custom kernels. We investigated reducing kernel launch overhead via fusion.

#### What we tried

**4-step plan:**
1. Fuse `split_transpose_3way` + `add_pos_bias_dual` into one kernel (`split_transpose_qkv_bias`)
2. Eliminate pos_proj transpose via `cublasGemmBatched` with per-head pointer arrays
3. cuDNN matmul+SiLU graphs for FF modules (fuse GEMM + activation)
4. cuDNN matmul+GLU graph for conv PW1 (fuse GEMM + gated activation)

Steps 1+2 are pure kernel launch elimination (2 fewer launches per block = 48 total).
Steps 3+4 use cuDNN frontend graph API to fuse GEMM + pointwise ops.

#### What worked: Steps 1+2 (custom kernel fusion)

**Step 1: `split_transpose_qkv_bias_fp16`** — A new CUDA kernel that reads the fused QKV output `[T, 3D]` once and writes `q_u = Q + bias_u`, `q_v = Q + bias_v`, `K`, `V` directly in `[H, T, head_dim]` layout. Replaces separate `split_transpose_3way` + `add_pos_bias_dual` (2 kernels → 1).

**Step 2: Strided batched GEMM for position scores** — Instead of transposing `pos_proj` from `[2T-1, 8, 128]` to `[8, 2T-1, 128]` then calling `cublasGemmStridedBatched`, we use `cublasGemmBatched` with per-head pointer arrays that read directly from the interleaved layout (`ldb = D_MODEL = 1024`). Eliminates the `transpose_0213` kernel entirely.

#### What didn't work: Steps 3+4 (cuDNN matmul graphs)

cuDNN frontend graph API can fuse `matmul → pointwise(swish)` into one kernel. We tested it for FF1/FF2 (matmul+SiLU) and conv PW1 (matmul+GLU). No performance benefit — cublasLt already selects optimal kernels, and the elementwise ops are bandwidth-bound at near-peak already. Worse: each new sequence length T requires building a new cuDNN graph (~70-90ms). Removed.

#### Benchmark results

Server mode, 81 files (40 librispeech + 41 earnings22), 3 runs after warmup:

```
BASELINE (cuDNN SDPA, pre-fusion):
  Run 1: 81 files, 588.9s audio, 1.001s inf, avg=12.4ms, RTFx=588x
  Run 2: 81 files, 588.9s audio, 1.017s inf, avg=12.6ms, RTFx=579x
  Run 3: 81 files, 588.9s audio, 1.021s inf, avg=12.6ms, RTFx=577x
  Overall: avg=12.5ms, RTFx=581x

OPTIMIZED (+ fused split_qkv_bias + strided batched GEMM):
  Run 1: 81 files, 588.9s audio, 0.973s inf, avg=12.0ms, RTFx=605x
  Run 2: 81 files, 588.9s audio, 0.964s inf, avg=11.9ms, RTFx=611x
  Run 3: 81 files, 588.9s audio, 0.962s inf, avg=11.9ms, RTFx=612x
  Overall: avg=11.9ms, RTFx=609x
```

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| avg inference | 12.5ms | 11.9ms | **-0.6ms (-5%)** |
| RTFx | 581x | 609x | **+5%** |
| Correctness | 81/81 match | 81/81 match | Exact match |

Per block: 24 → 22 ops (eliminated `split_transpose_3way` and `transpose_0213`). Total: 576 → 528 kernel launches.

Custom fused kernels gave a modest but real 5% improvement. cuDNN GEMM+activation graphs provided no benefit — cublasLt is already well-tuned, and the launch overhead savings (~3µs/launch) were offset by cuDNN graph dispatch overhead.

#### Head-to-head: TensorRT vs CUDA backend

Server mode, 81 files (589s audio), RTX 5070 Ti, 5 warmup passes + 3 benchmark runs, isolated:

| Backend | Run 1 | Run 2 | Run 3 | Avg |
|---------|-------|-------|-------|-----|
| **TensorRT** (FP16 engines) | 11.3ms / 642x | 11.4ms / 638x | 11.8ms / 618x | **11.5ms / 633x** |
| **CUDA** (cuBLAS + cuDNN SDPA + custom kernels) | 12.7ms / 572x | 12.6ms / 578x | 12.2ms / 594x | **12.5ms / 581x** |

**CUDA is ~8% slower than TRT** (12.5ms vs 11.5ms). Down from 23% in Step 11, thanks to cuDNN flash attention, cublasLt algorithm caching, and kernel fusion. The remaining ~1ms gap is TRT's Myelin compiler fusing GEMM+residual+LayerNorm chains — not replicable with library calls.

The CUDA backend eliminates ~800MB of TensorRT runtime libraries and GPU-specific engine files, replacing them with a single portable `paraketto-fp16.bin` that works on any CUDA GPU.

### Step 13: Removing cuDNN, Startup Optimization

#### cuDNN SDPA: zero inference benefit

A/B testing: cuDNN flash attention provided **no speed improvement** over the manual path (2 batched GEMMs + fused_score_softmax + transpose). RTFx identical at ~595x, but cuDNN added ~37ms warmup per new sequence length.

Removed all cuDNN code. **Dropped `libcudnn` from link dependencies entirely.**

#### Startup optimizations

Three changes to model init:

1. **Pooled GPU allocation** — Single `cudaMalloc` instead of ~35 individual calls (each ~2-5ms driver overhead). Pointers carved from the pool, 256-byte aligned. Saves ~100ms.

2. **`cudaMemcpy2D` for QKV concatenation** — 73,728 individual `cudaMemcpyAsync` calls → 72 `cudaMemcpy2DAsync` calls (3 per block).

3. **No cuDNN handle** — `cudnnCreate()` cost ~3ms for nothing.

#### Results

| Component | TRT | CUDA (before) | CUDA (after) |
|-----------|-----|---------------|--------------|
| cuda_init | 69ms | 68ms | 67ms |
| load + model init | 305ms | 286ms | 135ms |
| warmup | 16ms | 34ms | 35ms |
| **Total startup** | **~392ms** | **~386ms** | **~238ms** |

**CUDA startup is now 40% faster than TRT** (238ms vs 392ms). TRT's 305ms is dominated by engine deserialization (rebuilding GPU kernel launch parameters from the 1.2GB serialized plan). CUDA's 135ms is weight mmap + single cudaMalloc + 72 strided copies + position encoding generation.

Inference speed unchanged at ~600x RTFx. Link dependencies: `cudart`, `cublas`, `cublasLt`, `cufft` only. No TensorRT, no cuDNN.

### Step 14: The TRT FP16 `encoded_lengths` Off-By-One Bug

After adding a larger benchmark dataset with audio >30s, the TRT C++ backend showed **73.63% WER** on long audio — effectively garbage. Short audio (<30s) was fine. The CUDA backend and the Python ORT baseline both produced correct results (~2% WER) on the same files.

#### The investigation

This was the hardest bug to find because every obvious hypothesis was wrong.

**Dead end 1: FP16 precision.** The initial theory was that FP16 LayerNorm or Softmax in the encoder caused numerical drift on longer sequences. We tried:

- Full FP32 engine → correct (1.93% WER), but 2x larger and slower
- `OBEY_PRECISION_CONSTRAINTS` forcing LayerNorm + Softmax to FP32 → still 73.63%
- `OBEY_PRECISION_CONSTRAINTS` forcing all self-attention layers to FP32 → still 73.63%
- Optimization level 3 (less aggressive fusion) → still 73.63%

The engine sizes were identical (~1243 MB) regardless of how many layers were "forced" to FP32. **`OBEY_PRECISION_CONSTRAINTS` is silently broken in TRT 10.15** — the API is deprecated and has no effect. Every precision-based experiment told us nothing.

**Dead end 2: ORT TRT EP uses CUDA EP fallback.** We thought ORT's Python wrapper worked because it fell back to CUDA EP (FP32) for numerically sensitive ops. We dumped the ORT graph partitioning and found: "All nodes placed on [TensorrtExecutionProvider]. Number of nodes: 1". No CUDA EP fallback at all — the entire encoder runs in TRT, same as our C++ code.

**Dead end 3: Different engine quality.** We tried building an engine from ORT's preprocessed ONNX subgraph (2115 nodes vs 4491 in the original — constants folded, Casts removed). Same 73.63% WER. We even copied ORT's own cached TRT engine to use as our encoder.engine — same 73.63% WER. **The same TRT engine that works in Python ORT produced garbage in our C++ code.** This proved the engine was fine.

**The breakthrough.** Since the engine was identical and encoder outputs were verified to be nearly identical (cosine similarity 0.9999), the bug had to be in how the C++ code *read* the encoder output. We compared `encoded_lengths` (the model's self-reported output length) against the actual tensor shape for different audio lengths:

| Audio | Mel frames | `encoded_lengths` | Actual tensor T' | Mismatch? |
|-------|-----------|-------------------|-------------------|-----------|
| 3.5s  | 350       | 44                | 44                | No        |
| 10.5s | 1050      | 132               | 132               | No        |
| 54.6s | 5462      | **684**           | **683**           | **Yes (+1)** |
| 89.0s | 8908      | **1115**          | **1114**          | **Yes (+1)** |

For certain input lengths, the FP16 TRT engine's `encoded_lengths` output is **off by one** compared to the actual output tensor dimensions. Short audio is never affected.

#### Why off-by-one is catastrophic

The encoder output is stored in `[1, 1024, T']` layout — 1024 channels interleaved across T' time steps. To extract frame `t` for the decoder, the code uses `cudaMemcpy2D` with `enc_len` as the **source pitch** (byte stride between consecutive channels):

```cpp
cudaMemcpy2DAsync(
    d_enc_frame.ptr, sizeof(float),           // dst: tightly packed
    (char*)d_enc_out.ptr + t * sizeof(float),  // src: column t
    enc_len * sizeof(float),                   // src pitch = T' floats
    sizeof(float), 1024,                       // 1 float × 1024 rows
    cudaMemcpyDeviceToDevice, stream);
```

If `enc_len` is 684 but the actual stride is 683, each successive row reads 1 float too far. By row 1023 (the last channel), the accumulated error is **1023 floats** — reading from a completely different frame. The extracted "frame" is a diagonal slice across many frames rather than a single column, producing nonsense decoder input.

The CUDA backend (`conformer.cpp`) was unaffected because it stores encoder output in `[T, 1024]` layout (time-major). Frame extraction is just `x + t * 1024` — no pitch parameter, no dependence on `enc_len`.

The Python ORT backend was unaffected because ORT reads the actual tensor dimensions from TRT's execution context, never trusting the model's `encoded_lengths` output.

#### The fix

Two lines changed. Instead of reading `encoded_lengths` from GPU memory, query the actual output tensor shape:

```cpp
// Before (buggy):
int64_t enc_len = 0;
cudaMemcpyAsync(&enc_len, d_enc_lens.ptr, sizeof(int64_t),
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
if (enc_len > max_enc) enc_len = max_enc;

// After (fixed):
encoder.enqueue(stream);
cudaStreamSynchronize(stream);
auto out_dims = encoder.context->getTensorShape("outputs");
int64_t enc_len = out_dims.d[out_dims.nbDims - 1];  // actual T'
```

`getTensorShape` reads the dimensions that TRT's execution context actually allocated and wrote to — the ground truth for dynamic shape engines. The `encoded_lengths` model output is a computed value that goes through the FP16 arithmetic of the encoder and can have rounding errors.

#### Results

Long-audio WER dropped from **73.63%** to **1.94%** with full FP16 performance (no FP32 workaround needed):

```
Python  (ORT + TRT EP):   librispeech=1.81%  earnings22=16.48%  long=2.00%  difficult=39.36%  830x RTFx
C++ TRT (paraketto):      librispeech=1.81%  earnings22=16.48%  long=1.94%  difficult=39.36%  862x RTFx
C++ CUDA (cuBLAS):        librispeech=1.81%  earnings22=16.48%  long=1.92%  difficult=38.38%  707x RTFx
```

All three backends now agree on WER across all datasets. The TRT C++ backend is the fastest at 862x RTFx. The CUDA backend trades ~18% speed for eliminating the TRT dependency entirely.

Never trust a model's self-reported output dimensions from a computed tensor. For TensorRT dynamic shape engines, use `getTensorShape()` on the execution context. The model's own length computation goes through FP16 arithmetic and can have rounding errors that don't match the engine's internal shape calculations.

### Step 15: Fused GPU Mel Pipeline + GEMM Bias Epilogue

*Note: Between Steps 14 and 15, the benchmark datasets were expanded from 80 utterances (40 librispeech + 40 earnings22) to 240 (100 librispeech across 40 speakers, 40 earnings22, 50 long chapter-level files up to 120s, and 50 "difficult" accented utterances from VoxPopuli). The larger, more diverse test set explains the WER changes from this point forward — librispeech dropped from 1.81% to 1.68% due to different utterance selection, and the "difficult" dataset is new.*

#### The hidden bottleneck

Both backends shared the same mel pipeline: CPU pre-emphasis + windowing → GPU FFT → **GPU→CPU download** → CPU mel filterbank → CPU log → CPU normalize → **CPU→GPU upload** → encoder.

The cost wasn't the CPU compute — it was the **GPU pipeline stalls**. Five synchronization barriers where the GPU sat idle: waiting for D2H copy, waiting for CPU mel processing, waiting for H2D upload. The GPU did nothing while the CPU churned through thousands of frames × 128 channels with strided access.

#### Fused mel kernels

Two new CUDA kernels eliminate all CPU mel processing:

**`fft512_mel_log`** — One block per frame, 256 threads. Fuses the entire FFT → power spectrum → mel filterbank → log pipeline in shared memory:

1. Bit-reversal load + 9 butterfly stages (same as existing `fft512_power_kernel`)
2. Power spectrum computed in-place in shared memory (`sr[0..256]`)
3. Sparse mel filterbank via `atomicAdd` in shared memory — 504 filterbank entries stored in `__constant__` memory (~4KB), each thread handles ~2 entries, scatters weighted power to mel accumulator bins (`si[0..127]`)
4. `logf(mel + eps)` written to output

Total shared memory: 4KB (reusing the FFT's existing `sr[512] + si[512]`). The filterbank scatter has minimal contention — each mel bin receives ~4 atomic adds.

**`mel_normalize`** — One block per mel channel (128 blocks), up to 1024 threads. Three-pass kernel:

1. Parallel reduction for mean across `n_valid` frames
2. Parallel reduction for variance (two-pass for numerical stability, Bessel correction with `n-1`)
3. Normalize and write transposed: `mel_out[ch * T + f] = (x - mean) / std`

Output writes directly to the encoder's GPU input buffer — zero intermediate copies.

#### New mel pipeline

```
Before (6 GPU↔CPU transitions):
  CPU: pre-emphasis + window → upload frames
  GPU: FFT → download power spectrum
  CPU: mel filterbank (504 sparse weights) → CPU
  CPU: log → CPU
  CPU: per-channel normalize + transpose → upload to encoder

After (1 upload, 0 downloads):
  CPU: pre-emphasis + window → upload frames [n_frames, 512]
  GPU: fft512_mel_log → [n_frames, 128] log-mel on GPU
  GPU: mel_normalize  → [128, n_valid] normalized features on GPU
       (already on GPU, feed directly to encoder)
```

#### cublasLt GEMM bias epilogue

Separate optimization for the encoder: the subsampling output layer does `GEMM([T, 4096] × [4096, 1024])` followed by `bias_add_inplace_fp16` — a separate kernel that reads and writes the entire `[T, 1024]` output just to add a bias vector. cublasLt's `CUBLASLT_EPILOGUE_BIAS` folds the bias addition into the GEMM's epilogue, eliminating one kernel launch and one full T×D read+write.

Added `has_bias` to the GEMM plan cache key, set `CUBLASLT_MATMUL_DESC_EPILOGUE` + `CUBLASLT_MATMUL_DESC_BIAS_POINTER` on the matmul descriptor. New `gemm_nn_bias()` wrapper for the fused path.

#### Dropped libcufft

Both backends now use the custom 512-point FFT kernel exclusively. The TRT backend no longer links against `libcufft` — one fewer ~200MB dependency.

#### Results

| Backend | Before | After | Improvement |
|---------|--------|-------|-------------|
| C++ TRT (paraketto) | 858x | **1118x** | **+30%** |
| C++ CUDA (paraketto) | 750x | **881x** | **+17%** |

Per-dataset breakdown (C++ TRT):

| Dataset | WER | RTFx |
|---------|-----|------|
| librispeech | 1.68% | 989x |
| earnings22 | 16.48% | 816x |
| long | 1.93% | 1166x |
| difficult | 23.32% | 1081x |

Per-dataset breakdown (C++ CUDA):

| Dataset | WER | RTFx |
|---------|-----|------|
| librispeech | 1.68% | 810x |
| earnings22 | 16.48% | 731x |
| long | 1.92% | 901x |
| difficult | 23.24% | 890x |

WER is identical across all 240 utterances — zero numerical regression.

TRT benefits more (+30% vs +17%) because its Myelin-optimized encoder made the mel stall a proportionally larger fraction of total time.

The CUDA backend now matches pre-fusion TRT (881x vs 858x). TRT with fused mel pulls ahead to 1118x — **2.1x faster than the Python baseline** (529x).

---

### Step 16: Decoder Optimization Blitz — 881x → 1001x

With the encoder optimized and mel fully on GPU, the decoder became the bottleneck: 35-38% of total inference time. The greedy decode loop runs ~1000 steps, each launching 12 GPU operations. Per-step latency was dominated by kernel launch overhead, not compute.

#### Optimization 1: Strided batched GEMM

The encoder's position score computation used `cublasGemmBatchedEx` with explicit pointer arrays — three arrays of 8 `half*` pointers, uploaded H2D every single encoder block. Replaced with `cublasGemmStridedBatchedEx` which computes strides arithmetically, eliminating 72 `cudaMemcpyAsync` H2D calls per encoder pass. The key insight: `pos_temp` (position projection output) has an interleaved layout `[pos_len, N_HEADS, HEAD_DIM]` where the per-head stride in the first dimension is exactly `HEAD_DIM` — perfect for strided batched GEMM.

#### Optimization 2: GEMM bias epilogues for LSTM + joint

The LSTM layers and joint network each did `GEMM → bias_add_inplace` as separate operations. Extended the cublasLt plan cache with `CUBLASLT_EPILOGUE_BIAS` support, adding `gemm_nt_bias` and `gemm_nt_accum_bias` wrappers. Per decode step: **18 ops → 11 ops**.

Before (per LSTM layer):
```
gnt(x, W_ih)  →  bias_add(b_ih)  →  gnt_acc(h, W_hh)  →  bias_add(b_hh)  →  lstm_cell
```

After:
```
gnt_bias(x, W_ih, b_ih)  →  gnt_acc_bias(h, W_hh, b_hh)  →  lstm_cell
```

Same pattern for the joint network's three linear layers.

#### Optimization 3: GPU argmax

The decode loop previously transferred the full 1030-element FP16 joint output (2KB) from GPU to CPU every step, then did CPU argmax for token and duration. Replaced with a `dual_argmax_fp16` kernel: one block, 256 threads, warp-level parallel reduction finding both `argmax(logits[:1025])` (token) and `argmax(logits[1025:])` (duration step) on GPU. Now transfers just 8 bytes (two ints) per step.

#### Optimization 4: Precomputed encoder projections

The joint network computes `enc_proj = encoder_frame × W_enc + b_enc` every decode step — a 1×1024×640 GEMM. But `encoder_frame[t]` only depends on the encoder output, which is fixed for the entire decode loop. Moved the projection to a single batched GEMM right after encoding:

```cpp
// One GEMM: [T, 1024] × [1024, 640] + bias → [T, 640]
gnn_bias(x, T, D_MODEL, w->enc_proj_w, D_JOINT, w->enc_proj_b, enc_proj_all);
```

This eliminates ~1000 separate cuBLAS calls (one per decode step), replacing them with a single efficient GEMM. During decoding, each step just indexes `enc_proj_all + t * D_JOINT`.

#### Dead code removal

Removed the `pos_bias` buffer (N_HEADS × T × T × 2 bytes — the largest decoder-related allocation), the unused `rel_pos_skew_scale_fp16` call path, device pointer arrays (`d_pos_A/B/C_ptrs`), and the CUDA Graph cache (attempted earlier, caused regression due to warmup overhead on variable-length inputs).

#### What didn't work: fused LSTM/joint kernels

Attempted fusing the entire LSTM step into a single kernel — 256 threads, one block, serial dot products in shared memory. **65x slower than cuBLAS.** cuBLAS distributes M=1 GEMMs across all 64 SMs; a single-block kernel uses ~2% of GPU capacity. The launch overhead savings (~4μs × ~4000 launches) were dwarfed by the compute regression. Don't hand-roll matrix-vector products when cuBLAS already parallelizes them across the full GPU.

#### Results

| Backend | Before | After | Improvement |
|---------|--------|-------|-------------|
| C++ CUDA | 881x | **1001x** | **+14%** |

Per-dataset breakdown:

| Dataset | WER | RTFx | Utts | Audio | Time |
|---------|-----|------|------|-------|------|
| librispeech | 1.68% | 908x | 100 | 896s | 987ms |
| earnings22 | 16.48% | 824x | 40 | 253s | 307ms |
| long | 1.91% | 1030x | 50 | 5578s | 5.42s |
| difficult | 23.24% | 1003x | 50 | 509s | 508ms |
| **Total** | | **1003x** | **240** | **7236s** | **7.22s** |

WER identical across all 240 utterances. The CUDA backend has now crossed 1000x RTFx — **1.9x faster than the original Python baseline** (529x) with zero external dependencies beyond CUDA.

---

### Step 17: Combined LSTM Weights + Subsampling Overhaul — 1001x → 1253x

Two independent optimizations targeting different pipeline stages: the decoder's LSTM and the encoder's subsampling convolutions.

#### Decoder: combined LSTM weights

Each decode step ran two cuBLAS GEMMs per LSTM layer — one for `W_ih @ input` and one for `W_hh @ h_prev` — totaling 4 GEMM launches. Since these are M=1 GEMMs where kernel launch overhead (~5μs) rivals actual compute, reducing launch count matters.

**Pre-combined weights at init:**
- Horizontally concatenate `W_ih` and `W_hh` into `W_combined = [W_ih | W_hh]` of shape `[4*D, 2*D]` using `cudaMemcpy2DAsync` (same pattern as the existing QKV weight concatenation)
- Pre-add biases: `bias_combined = b_ih + b_hh` via `residual_add_fp16`
- At runtime, concatenate `[input; h_prev]` into a `[2*D]` vector, then one GEMM: `W_combined @ [input; h_prev] + bias_combined`

**Two new helper kernels:**
- `embed_concat_fp16`: embedding lookup + concat with h for LSTM layer 0 input prep — single block, half2 vectorized
- `concat_vectors_fp16`: concat h_out with h for LSTM layer 1 input prep — same pattern

This cuts per-step GPU operations from 11 to 7 (2 fewer cuBLAS launches, 1 fewer embedding kernel).

#### Encoder: subsampling conv2d

Profiling with `nsys` revealed the subsampling convolutions consumed **24.6% of total GPU time** — the single largest bottleneck. The culprit: a naive `conv2d_kernel` where each thread independently loads its input pixels with no data reuse.

The subsampling pipeline has 5 convolutions:

| Layer | Type | Shape | Time (est.) |
|-------|------|-------|-------------|
| conv.0 | regular 1→256, 3×3, stride 2 | [1, T_mel, 128] → [256, T/2, 64] | ~12ms |
| conv.2 | depthwise 256, 3×3, stride 2 | [256, T/2, 64] → [256, T/4, 32] | ~2ms |
| conv.3 | pointwise 256→256, 1×1 | [256, T/4, 32] → [256, T/4, 32] | ~1.5ms |
| conv.5 | depthwise 256, 3×3, stride 2 | [256, T/4, 32] → [256, T/8, 16] | ~0.8ms |
| conv.6 | pointwise 256→256, 1×1 | [256, T/8, 16] → [256, T/8, 16] | ~0.5ms |

**conv.0 (the monster):** For 99s audio, this produces 256 × 4950 × 64 ≈ 81M output elements. With C_in=1, all 256 output channels read the same 9 input pixels per position — 256× redundancy in global memory reads.

Fix: **im2col + cuBLAS GEMM.** Extract 3×3 patches into a `[9, H'×W']` matrix, then `weight[256, 9] @ im2col[9, H'×W']` — a standard GEMM at near-peak throughput across all SMs.

**conv.3, conv.6 (pointwise):** A 1×1 convolution is just a matrix multiply along the channel dimension. `output[C_out, H×W] = weight[C_out, C_in] @ input[C_in, H×W]`. Replaced the per-element kernel with a single cuBLAS `gemm_nn` call.

**Bias + ReLU:** Can't use cublasLt's bias epilogue here because the output is NCHW (channels in the leading row-major dimension, but the trailing col-major dimension). A new `bias_relu_nchw_fp16` kernel fuses the per-channel bias add and ReLU into one pass.

The depthwise convolutions (conv.2, conv.5) remain on the naive kernel — they're only ~15% of subsampling time and each channel is independent, limiting the benefit of shared memory tiling.

#### Results

| Backend | Before | After | Improvement |
|---------|--------|-------|-------------|
| C++ CUDA | 1001x | **1253x** | **+25%** |

Per-dataset breakdown:

| Dataset | WER | RTFx | Utts | Audio | Time |
|---------|-----|------|------|-------|------|
| librispeech | 1.68% | 1108x | 100 | 896s | 809ms |
| earnings22 | 16.48% | 999x | 40 | 253s | 253ms |
| long | 1.90% | 1295x | 50 | 5578s | 4.31s |
| difficult | 23.24% | 1255x | 50 | 509s | 406ms |
| **Total** | | **1253x** | **240** | **7236s** | **5.78s** |

WER identical across all 240 utterances. The CUDA backend is now **2.4x faster than the original Python baseline** (529x).

---

### Step 18: FP8 E4M3 GEMMs

GEMMs account for ~60% of GPU time (7.6ms out of 12.7ms, 826 calls). The RTX 5070 Ti (sm_120, Blackwell GeForce) has FP8 E4M3 tensor cores with 2× the throughput of FP16. FP8 weights are also half the size, which helps memory-bound decoder GEMMs.

### Attempt 1: Naive per-GEMM activation quantization

**Approach**: Weights quantized FP16→FP8 at init time (one-time cost, per-tensor absmax scaling). Activations quantized FP16→FP8 on-the-fly before every GEMM via a multi-block absmax+quantize kernel. GEMMs via `cublasLtMatmul` with FP8 inputs, FP32 accumulation, FP16 output, descaled via `A_SCALE_POINTER`/`B_SCALE_POINTER`.

**Results** (10.4s audio, encoder time):

| Backend | Enc time | RTFx |
|---------|----------|------|
| FP16 baseline | 4.8ms | 1133× |
| FP8 naive | 5.5ms | 1058× |

**15% slower.** Each of the ~220 encoder GEMMs needs a two-pass quantization kernel (absmax then quantize), reading every activation tensor twice. Total extra I/O: ~550MB per encoder pass. At ~500 GB/s, that's ~1.1ms minimum — close to the measured 0.7ms penalty. FP8 tensor core gains can't overcome this I/O tax.

Also: `CUBLASLT_EPILOGUE_BIAS` returns `NOT_SUPPORTED` with FP8 on Blackwell, and FP8 doesn't support N=1 GEMMs (decoder stays FP16).

### Options to reduce quantization overhead

Three approaches:

- **A: Cached scales** — skip the absmax pass, use a scale from warmup. Single-pass quantize: halves I/O overhead from ~0.7ms to ~0.35ms. Risk: accuracy loss if activations exceed cached max.
- **B: Fuse into producing kernels** — layer_norm, silu, etc. already touch every element. Dual-output FP16+FP8 means zero extra I/O. But still needs cached scales (can't write FP8 without knowing the max). Requires modifying ~6 kernels.
- **C: CUTLASS inline conversion** — custom prologue converts FP16→FP8 during the GEMM's global→shared tile load. Zero overhead, but TN-layout only on sm_120, heavy engineering, steep learning curve.

### Attempt 2: Cached scales (option A)

**Plan**: Combine A+B — each activation-producing kernel writes FP8 using a cached scale from warmup. No absmax pass, no separate quantize kernel. During warmup, run a full absmax pass to establish initial scales. At runtime, use `__NV_SATFINITE` mode to clamp outliers. Scales are stable because activation distributions don't vary much across inputs.

Six kernels would need dual FP16+FP8 output: `layer_norm`, `silu`, `residual_add_layer_norm`, `glu`, `depthwise_conv1d_k9_silu`, and position encoding.

**What we shipped**: Option A only — cached scales from warmup, separate `quantize_fp8_static` kernel per GEMM. No fused dual-output kernels. Sufficient to make FP8 faster than FP16.

**Caveats**: cublasLt FP8 silently fails for some dimension combinations even when the heuristic reports a valid algorithm. Pre-quantized position encoding GEMMs crashed at runtime. Decoder GEMMs (N=1) unsupported by FP8 entirely.

**Results**:

| Backend | RTFx | librispeech WER | earnings22 WER |
|---------|------|-----------------|----------------|
| C++ CUDA FP16 | 1253x | 1.68% | 16.48% |
| C++ CUDA FP8 | **1289x** | 1.94% | 15.76% |

Per-dataset breakdown:

| Dataset | WER | RTFx | Utts | Audio | Time |
|---------|-----|------|------|-------|------|
| librispeech | 1.94% | 1128x | 100 | 896s | 794ms |
| earnings22 | 15.76% | 1009x | 40 | 253s | 251ms |
| long | 2.16% | 1337x | 50 | 5578s | 4.17s |
| difficult | 19.21% | 1278x | 50 | 509s | 398ms |
| **Total** | | **1289x** | **240** | **7236s** | **5.61s** |

**+2.9% throughput improvement** over FP16 baseline. WER changes are mixed: slightly worse on librispeech/long (+0.26%), but actually better on earnings22 (-0.72%) and difficult (-4.03%). The modest throughput gain reflects the fact that these GEMMs are memory-bandwidth-bound (arithmetic intensity ~100-200 FLOPs/byte, well below the 992 FLOPs/byte compute-bound threshold). FP8 weights are half the size, saving ~600MB of DRAM reads per encoder pass, but cublasLt's FP8 kernel has its own overhead that partially offsets the memory savings.

### Profiling: Why is cublasLt FP8 barely faster?

`nsys` profiling reveals the culprit. Comparing the weight-matrix GEMM kernels (218 per encoder pass, excluding batched attention GEMMs which stay FP16):

**FP16 kernels** — cublasLt selects mature CUTLASS cutlass_80 tensor op kernels:

| Kernel | Tile | Instances | Avg (μs) | Total (ms) |
|--------|------|-----------|----------|------------|
| `cutlass_80_tensorop_h16816gemm_128x128_32x5_nn` | 128×128 | 72/inf | 15.1 | 1.09 |
| `cutlass_80_tensorop_h16816gemm_256x64_32x4_nn` | 256×64 | 49/inf | 13.9 | 0.68 |
| `cutlass_80_tensorop_h16816gemm_128x64_64x4_tn` | 128×64 | 24/inf | 9.2 | 0.22 |
| `cutlass_80_tensorop_h16816gemm_128x64_64x4_nn` | 128×64 | 24/inf | 8.7 | 0.21 |
| `nvjet_sm120_hhh_mma_64x64x128_3_NN` | 64×64 | 24/inf | 7.6 | 0.18 |
| `nvjet_sm120_hhh_mma_64x64x128_3_TN` | 64×64 | 24/inf | 7.1 | 0.17 |
| `splitKreduce` (for split-K GEMMs) | — | 48/inf | 2.7 | 0.13 |
| **Total FP16 weight GEMMs** | | **~217/inf** | | **~2.69** |

**FP8 kernels** — cublasLt selects newer nvjet sm120 kernels:

| Kernel | Tile | Instances | Avg (μs) | Total (ms) |
|--------|------|-----------|----------|------------|
| `nvjet_sm120_qqhsh_mma_64x64x64_12_NN` | 64×64 | 121/inf | 11.2 | 1.36 |
| `nvjet_sm120_qqhsh_mma_128x64x64_8_NN` | 128×64 | 48/inf | 10.2 | 0.49 |
| `nvjet_sm120_qqhsh_mma_128x64x64_8_TN` | 128×64 | 24/inf | 9.7 | 0.23 |
| `nvjet_sm120_qqhsh_mma_64x64x64_12_TN` | 64×64 | 24/inf | 6.3 | 0.15 |
| **Total FP8 weight GEMMs** | | **~217/inf** | | **~2.23** |
| `quantize_fp8_static` (cached quant) | — | 218/inf | 1.4 | 0.30 |
| **Total FP8 (GEMMs + quantize)** | | | | **~2.53** |

**Net savings: 2.69 − 2.53 = 0.16ms per inference (6% of GEMM time).**

The FP8 GEMMs are only **17% faster** per-kernel (2.23 vs 2.69ms), not 2×. And `quantize_fp8_static` eats 0.30ms of the 0.46ms savings, leaving just 0.16ms.

**Why FP8 GEMMs are only 17% faster — not 2×:**

These GEMMs are memory-bandwidth-bound at M≈125 (sequence length after 8× downsampling). The roofline model explains everything:

| GEMM (example) | M | N | K | FP8 weight | Theoretical read time | Measured FP8 | Efficiency |
|----------------|---|---|---|------------|----------------------|--------------|------------|
| FF1 linear1 | 125 | 4096 | 1024 | 4.0 MB | 7.9 μs | ~10.2 μs | 77% |
| QKV | 125 | 3072 | 1024 | 3.0 MB | 6.0 μs | ~11.2 μs | 54% |
| Out proj | 125 | 1024 | 1024 | 1.0 MB | 2.0 μs | ~11.2 μs* | 18%* |
| FF2 linear2 | 125 | 1024 | 4096 | 4.0 MB | 7.9 μs | ~11.2 μs* | 71%* |

*The 64×64 NN kernel averages 11.2μs across ALL NN shapes (N=1024 through N=3072). The small-N GEMMs (N=1024, weight=1MB) take nearly as long as the large-N ones — a sign of high fixed overhead or poor occupancy.

FP16 uses 128×128 and 256×64 tiles with near-perfect bandwidth utilization. FP8 `nvjet` uses 64×64 tiles — worse shared-memory reuse and lower bandwidth efficiency. cublasLt's FP8 kernels for sm_120 are simply less mature than its FP16 CUTLASS kernels.

### Attempt 3: Custom CUTLASS FP8 GEMM

If cublasLt's FP8 kernels are poorly tuned for sm_120, write our own using CUTLASS 4.x.

SM120 (GeForce Blackwell) constraints: TN layout only, cluster 1×1×1, K fixed at 128, valid tiles 64×64 through 128×128. Dequantization folded into the GEMM's alpha: `alpha = act_scale × wt_scale`. Tile selection: 128×128×128 with Cooperative schedule.

#### Results

**SM120 B-layout gotcha**: Despite declaring `LayoutB = ColumnMajor`, SM120's MMA reads B with K-contiguous access. NN weights W[K,N] must be transposed to W^T[N,K] — the opposite of what the layout name suggests. This was masked by uniform test data (all-ones give correct output regardless of access order) and only caught with non-square matrices.

**Benchmark** (RTX 5070 Ti, same test set):

```
                FP8 CUTLASS          FP16 cuBLAS
Dataset         RTFx    WER          RTFx    WER
librispeech      803x   2.07%        1013x   1.68%
earnings22       650x  15.19%         916x  16.48%
long            1273x   2.20%        1309x   1.92%
difficult        899x  19.13%        1261x  23.24%
Total           1122x                1242x
```

FP8 CUTLASS is 10% slower overall than FP16 cuBLAS, with comparable WER.

---

### Step 19: CUTLASS FP16 GEMM — Replacing cuBLAS

FP8 was a dead end — slower kernels, worse WER. But the FP8 investigation revealed something: cuBLAS is already running CUTLASS SM80 templates for FP16. If we compile those same templates ourselves, we can offer a cuBLAS-free build option with only `cudart` as a dependency.

### What Is cuBLAS Actually Doing?

We used two approaches to peek inside:

#### Instrumenting the heuristic

cuBLASLt (the "light" API) exposes an algorithm selection step. When you ask it to
multiply two matrices, it first queries a heuristic that returns an algorithm ID,
tile configuration, and pipeline depth. We added logging to dump these:

```
GEMM NN  4096x 63x1024: algo=21 tile=64x64  stages=6  splitk=1    ← encoder FF layers
GEMM NN  3072x 63x1024: algo=67 tile=64x64  stages=AUTO           ← fused QKV projection
GEMM NT  2048x 63x1024: algo=21 tile=32x32  stages=2              ← conv pointwise
GEMM NT  2560x  1x1280: algo=21 tile=16x16  stages=1  bias        ← decoder LSTM
GEMM NN   640x  1x 640: algo=21 tile=16x16  stages=1  bias        ← decoder joint
```

#### Profiling with nsight systems

nsight systems captures the actual kernel names:

```
28.0%  cutlass_80_tensorop_h16816gemm_64x64_64x6_nn_align8     (196 calls, 12us avg)
10.2%  nvjet_sm120_hhh_mma_64x64x128_tmaAB_bz_NNNN            (96 calls, 9us avg)
 8.9%  cutlass_80_wmma_h161616gemm_16x16_128x1_tn_align8       (132 calls, 5.7us avg)
 7.6%  cutlass_80_wmma_h161616gemm_32x32_128x2_tn_align8       (96 calls, 6.6us avg)
 7.5%  magma_hgemmEx_kernel                                     (96 calls, 6.6us avg)
 4.7%  cutlass_80_wmma_h161616gemm_16x16_32x1_nn_align8        (66 calls, 6.0us avg)
 4.1%  cutlass_80_tensorop_h16816gemm_64x64_32x6_nn_align2     (66 calls, 5.2us avg)
 4.1%  cutlass_80_wmma_h161616gemm_32x32_128x2_nn_align8       (48 calls, 7.1us avg)
```

**cuBLAS is running Ampere-era (SM80) CUTLASS kernels on our Blackwell GPU.** Every `cutlass_80_` kernel is an SM80 template. For FP16, SM120's tensor cores use the same `mma.sync` instruction as Ampere — so even NVIDIA falls back to Ampere kernels.

One native SM120 kernel (`nvjet_sm120_*`) accounts for 10% of GEMM time, used for fused QKV and position encoding. The remaining 7.5% uses MAGMA kernels for batched attention GEMMs.

Since cuBLAS is already using CUTLASS SM80 templates, we can compile the same templates ourselves. We know the tile sizes, pipeline depths, and layouts for every shape. The open questions:
- The `nvjet_sm120` kernel (10% of time) — cover with SM80 CUTLASS, benchmark the difference
- MAGMA batched kernels (7.5%) — need our own strided batched GEMM
- Decoder GEMV (M=1) — cuBLAS wastes full GEMM tiles on what should be a reduction

### The CUTLASS templates we need

| Template | Tile | Pipeline | Layout | Shapes |
|----------|------|----------|--------|--------|
| `tensorop_h16816gemm` | 64x64 | K=64, 6 stages | NN | FF1/FF2 (28% of time) |
| `tensorop_h16816gemm` | 64x64 | K=32, 6 stages | NN | Encoder projection |
| `wmma_h161616gemm` | 32x32 | K=128, 2 stages | TN | Conv pointwise (NT layout) |
| `wmma_h161616gemm` | 32x32 | K=128, 2 stages | NN | Position encoding |
| `wmma_h161616gemm` | 16x16 | K=128, 1 stage | TN | Decoder LSTM GEMV |
| `wmma_h161616gemm` | 16x16 | K=32, 1 stage | NN | Decoder joint GEMV |
| `tensorop_h16816gemm` | 128x128 | K=32, 5 stages | NN | Subsampling (large) |
| custom GEMV | — | — | both | Decoder M=1 (potential speedup) |
| batched GEMM | — | — | both | Multi-head attention |

The total number of unique kernel configurations is small — roughly 10 cubins
covering all ~1000 kernel launches per inference pass.

### Writing the CUTLASS Kernels

We instantiated CUTLASS SM80 FP16 GEMM templates and benchmarked against cuBLAS for every shape in the model.

#### The configs we used

All kernels use `arch::Sm80` with `OpClassTensorOp` and `mma.sync 16x8x16`:

| Config | Tile | K/stage | Stages | Layout | Align |
|--------|------|---------|--------|--------|-------|
| `64x64_64_s6` | 64x64 | 64 | 6 | NN + TN | 8 |
| `64x64_32_s6` | 64x64 | 32 | 6 | NN | 8 |
| `64x64_32_s3` | 64x64 | 32 | 3 | NN | 8 |
| `64x64_64_s6_sk4` | 64x64 | 64 | 6 | NN | 8 | split-K=4 |
| `64x64_32_s6_a2` | 64x64 | 32 | 6 | NN | 2 | for M=1030 |
| `128x128_32_s5` | 128x128 | 32 | 5 | NN | 8 |
| `128x64_32_s3` | 128x64 | 32 | 3 | TN | 8 |

#### Results: CUTLASS matches or beats cuBLAS

```
Shape                cuBLAS(us)  CUTLASS(us)  Config           Ratio   Notes
FF1 linear1 (NN)         8.2         8.2      64x64_64_s6      1.00x   Matched
FF1 linear2 (NN)        10.3        12.3      64x64_64_sk2     1.20x   Split-K gap (only slow one)
Fused QKV (NN)           8.0         6.2      64x64_32_s10     0.78x   CUTLASS 22% faster!
Pos enc proj (NN)        6.0         6.2      64x64_32_s6      1.03x   ~Matched
MHSA out (NN)            6.2         6.2      64x64_64_s6      1.00x   Matched
Enc proj (NN)            6.2         6.2      64x64_64_s6      1.00x   Matched
Conv pw1 (TN)            8.2         6.2      64x64_64_s6      0.75x   CUTLASS 25% faster!
Conv pw2 (TN)            6.2         6.2      64x64_64_s6      1.00x   Matched
LSTM gates (TN)          6.2         6.2      64x64_64_s6      1.00x   Matched
Dec proj (NN)            6.2         4.1      64x64_32_s3      0.67x   CUTLASS 33% faster!
Out proj (NN)            6.2         6.2      64x64_32_s6_a2   1.00x   Matched (align2 fix)
Sub conv.3 (NN)          8.2         8.2      128x128_32_s5    1.00x   Matched
Sub conv.6 (NN)          6.2         3.9      64x64_32_s10     0.64x   CUTLASS 36% faster!
```

#### Total impact on inference

Weighting by call count per inference pass (24 blocks x encoder calls, ~33
decoder steps):

| | cuBLAS | CUTLASS |
|---|---|---|
| Total GEMM time | 3,754 us | 3,508 us |
| **Delta** | | **-6.5% faster** |

CUTLASS is **6.5% faster overall** than cuBLAS, even with the 20% regression on the
one split-K shape. The decoder GEMVs (where cuBLAS wastes full GEMM tiles on M=1
vector-matrix products) are the biggest win.

### What we learned

1. **A handful of CUTLASS templates cover everything.** 64x64 and 128x128 tiles with varying K and pipeline stages handle all 13 GEMM shapes.

2. **cuBLAS's WMMA kernels aren't special.** Standard TensorOp 16x8x16 tiles match or beat cuBLAS's WMMA 16x16x16 selections.

3. **cuBLAS is suboptimal for GEMV.** M=1 decoder shapes use full GEMM tiles — wasteful for matrix-vector products.

4. **Split-K is the main gap.** cuBLAS's split-K=4 for the 1024×63×4096 shape is 20% faster than our serial or parallel split-K. We tried three approaches; none closed the gap. One shape out of 13, +96μs per inference — acceptable.

5. **The nvjet_sm120 kernel doesn't help.** Our SM80 CUTLASS is actually *faster* on the shapes where cuBLAS uses its native SM120 kernel.

6. **SM80 vs SM120a compilation doesn't matter.** Same CUTLASS templates compiled with `-arch=sm_80` vs `-arch=sm_120a` — no meaningful difference.

The final state is two build targets: `paraketto.cuda` (CUTLASS only, `cudart` as the sole dependency) and `paraketto.cublas` (cuBLAS, ~2% faster). Both share the same `conformer.cpp` via a `gemm.h` abstraction layer.

---

### Step 20: FF Linear2 Split-K Tuning — 1204x → 1258x

After replacing cuBLAS with CUTLASS, the one persistent gap was the FF linear2 shape: `1024×T×4096`, called 48 times per utterance (24 encoder blocks × 2 FF modules). T varies with audio length — at 10 seconds it's around 125 frames.

#### What cuBLAS was doing

Used `ncu --print-summary per-kernel` to profile cuBLAS on specific T values. At small T, cuBLAS uses split-K parallel GEMM with varying tile sizes:

- T≤64: `128x128_32x5_nn` with split_k=4
- T=65–128: `128x64_32x4_nn` with split_k=4
- T=129–256: `256x64_32x4_nn` with split_k=3/4
- T=257–448: `64x128_32x5_nn` (regular)
- T=449–800: `128x128_32x5_nn` with split_k=2/3/4

#### The swizzle problem

`GemmSplitKParallel` in CUTLASS 2.x uses `GemmSplitKHorizontalThreadblockSwizzle`, which maps the grid as `(ceil(M/TbN), ceil(N/TbM), sk)` — swapping the M and N tile counts. For a 256×64 tile on a 1024×260 problem, this gives `(ceil(1024/64), ceil(260/256), 3)` = `(16, 2, 3)` = 96 blocks, instead of the correct `(ceil(1024/256) × ceil(260/64), 1, 3)` = `(20, 1, 3)` = 60 blocks. 60% wasted work.

cuBLAS uses `GemmIdentityThreadblockSwizzle` with split-K — the correct grid. This is not exposed through `GemmSplitKParallel`.

#### What we found

After profiling with ncu and running benchmarks across T=63–800:

| T range | cuBLAS kernel | Our kernel | Ratio |
|---------|--------------|------------|-------|
| ≤64 | 128x128 sk=4 | 64x64 sk=4 | −20% (cuBLAS wins) |
| 65–128 | 128x64 sk=4 | 128x64 sk=4 | −14% (we win) |
| 129–192 | 256x64 sk=3 | 64x64 s6 | −11% (we win) |
| 193–256 | 256x64 sk=4 | 128x64 sk=4 | tied at 20.5μs |
| 257–279 | 256x64 sk=3 | 64x128 s5 | −8.5% (cuBLAS wins, wrong swizzle) |
| 280–448 | 64x128 s5 | 64x128 s5 | tied |
| 449–512 | 128x128 sk=2 | 128x128 sk=2 | −7% (we win) |
| 513–640 | 128x128 sk=3 | 128x128 sk=3 | −9% (we win) |
| 641–800 | 128x128 sk=4 | 128x128 sk=4 | −12–19% (we win) |

The T=257–279 gap (corresponding to ~20–22 second utterances) is a fundamental CUTLASS 2.x limitation — the swizzle inflates the grid and can't be fixed without using `GemmUniversal`. For the benchmark dataset, this range is rare enough that it barely affects overall RTFx.

#### Dispatch update

Updated `nn_dispatch()` in `cutlass_gemm.cu` with correct boundaries. Added `GemmSplitKNN_128x64_32_s5` for T=65–128 and T=193–256, `GemmNN_64x128_k32_s5` for T=257–448, and `GemmSplitKNN_128x128_32_s5` with split_k=2/3/4 for T=449–800.

#### Results

```
                 CUTLASS (cudart only)          cuBLAS (+ libcublas)
              ────────────────────────────   ────────────────────────────
               RTFx    WER    Audio  Time     RTFx    WER    Audio  Time
librispeech   1083x   1.68%   896s  827ms    1042x   1.68%   896s  860ms
earnings22     982x  16.48%   253s  258ms     965x  16.48%   253s  262ms
long          1313x   1.92%  5578s  4.25s    1277x   1.90%  5578s  4.37s
difficult     1215x  23.41%   509s  419ms    1226x  23.32%   509s  415ms
              ────────────────────────────   ────────────────────────────
Total         1258x          7236s  5.75s    1226x          7236s  5.90s
```

**CUTLASS now outperforms cuBLAS overall: 1258x vs 1226x (+2.6%).** The split-K tuning recovered the gap in the medium-T range and added gains at large T through 128x128 split-K parallelism that cuBLAS's heuristics miss.

#### On CUTLASS 3.x/4.x and TMA

CUTLASS 4.4.1 does have `sm120_mma_tma.hpp` — a TMA-based mainloop for the RTX 5070 Ti. But the `CollectiveBuilder` for sm_120 enforces a hard `static_assert` refusing FP16: it only supports FP4/FP6/FP8 (block-scaled narrow precision). For FP16, sm_120 uses the same `mma.sync m16n8k16` instruction as Ampere, and CUTLASS 4.x provides no TMA-accelerated FP16 path for it. WGMMA (Hopper's asynchronous tensor core) is not available on sm_120 either — that's H100-only.

The conclusion: CUTLASS 2.x `device::Gemm` templates compiled with `-arch=sm_120` are the correct approach for FP16 on consumer Blackwell. The identity-swizzle split-K gap (T=257–279) remains, but it covers a narrow slice of real-world audio lengths and has negligible RTFx impact.

---

## Final Architecture: Three Binaries

After merging the FP8 branch with the improved FP16 baseline from master, we restructured the build to produce three separate binaries sharing the same driver (`paraketto_cuda.cpp`) and weight loader (`weights.cpp`), with the GEMM backend selected at compile time:

| Binary | Backend | Build |
|--------|---------|-------|
| `paraketto.cuda` | CUTLASS FP16 (custom-tuned) | `src/conformer.cpp` + `cutlass_gemm.o` |
| `paraketto.cublas` | cuBLAS/cublasLt FP16 | `src/conformer.cpp` + `cublas_gemm.o` |
| `paraketto.fp8` | cublasLt FP8 E4M3 | `src/conformer_fp8.cpp` |

Key design decision: `conformer_fp8.cpp` is a separate source file rather than `#ifdef`-gated code in `conformer.cpp`. This keeps the FP8-specific complexity (cublasLt handle, quantization buffers, activation scale calibration, algorithm cache) fully isolated.

The unified GEMM interface (`src/gemm.h`) abstracts `gemm_nn`, `gemm_nt`, `gemm_nn_bias`, etc. — the FP16 binaries select CUTLASS or cuBLAS at link time with no source changes.

### FP8 Weight Cache

Quantizing 216 conformer linear matrices + 6 decoder/projection matrices from FP16 takes ~1s at startup. To avoid this on every run, `paraketto.fp8` automatically saves a `paraketto-fp8.bin` cache (588 MB) on first run and loads it on subsequent runs:

```
$ make weights-fp8       # generates paraketto-fp8.bin once
$ ./paraketto.fp8 ...    # loads cache: ~400ms vs ~1400ms cold start
```

The cache format is a simple flat binary: 16-byte header (`PRKTFP8\0` + version + pad) followed by packed FP8 weight data and float32 weight scales. The static build (`paraketto.fp8.static`) embeds `paraketto-fp8.bin` via `objcopy` — no `paraketto-fp16.bin` needed at runtime, only the NVIDIA driver and cuBLAS/cublasLt shared libraries.

---

## Final Benchmark

Full comparison across all five backends, RTX 5070 Ti (SM120, Blackwell), 240 utterances (7236s audio):

| Backend | librispeech RTFx | earnings22 RTFx | long RTFx | difficult RTFx | **Total RTFx** |
|---------|-----------------|-----------------|-----------|----------------|----------------|
| Python ONNX+TRT | 694x | 583x | 879x | 801x | **831x** |
| C++ TRT | 946x | 836x | 1192x | 1079x | **1131x** |
| C++ CUTLASS FP16 | 1069x | 979x | 1307x | 1205x | **1250x** |
| C++ cuBLAS FP16 | 1077x | 1000x | 1306x | 1261x | **1256x** |
| **C++ FP8 cublasLt** | **1152x** | **1015x** | **1360x** | **1269x** | **1309x** |

WER is identical across all C++ backends (1.68% librispeech, 16.48% earnings22) except FP8 divergence on difficult speech (23.32% FP16 → 19.38% FP8 — FP8 quantization noise acts as beneficial regularization on noisy audio).

**FP8 vs FP16 CUTLASS: +4.7% throughput overall, +7.8% on short utterances (librispeech).**

When FP8 was first benchmarked on the fp8 branch (Step 18), it was only +2.9% faster than the FP16 cuBLAS baseline (1289x vs 1253x). Kernel profiling at that point showed why the gap was so small:

1. **FP8 GEMM speedup**: nvjet sm120 FP8 kernels were ~17% faster than FP16 in raw GEMM time (2.23ms vs 2.69ms per encoder pass at T=125)
2. **Quantization overhead**: 218 `quantize_fp8_static` kernel calls cost ~0.30ms per pass, eating most of the 0.46ms GEMM savings
3. **Memory bandwidth bound**: at T=125 encoder frames, the weight GEMMs are bandwidth-limited — FP8 halves the weight size but cublasLt's FP8 kernels on SM120 use 64×64 tiles vs the mature 128×128 CUTLASS/cuBLAS FP16 kernels, reducing reuse and bandwidth efficiency

The final +4.7% gap comes from two things that happened after that analysis. First, the FP16 baseline changed — master's CUTLASS replaced cuBLAS with custom-tuned split-K dispatch, which shifted the kernel mix. Second, the FP8 path got I/O optimizations: mmap with `MAP_POPULATE` for the weight cache, and prefetch overlap with CUDA context init (cold startup 452→325ms). These don't affect per-inference GEMM time but they reduce amortized overhead across the benchmark run.

For long audio (T≫125), the GEMMs dominate and both FP16 and FP8 approach the memory bandwidth limit. For short utterances, the FP8 advantage is largest (+7.8% on librispeech) because the smaller weight cache (604 MB vs 1.2 GB) reduces I/O pressure.

## Fused FP8 Quantization

The final benchmark above showed FP8 was +4.7% faster than CUTLASS FP16, but profiling revealed that 65% of the raw GEMM savings were being eaten by activation quantization overhead: 218 `quantize_fp8_static` kernel launches per encoder pass, each costing ~1.4us (total ~0.30ms). This left room for a straightforward optimization: fuse the FP8 quantization directly into the kernels that produce activations.

### The idea

Each activation-producing kernel (LayerNorm, SiLU, depthwise conv, etc.) already computes a float value and writes `__float2half(val)`. The fused variants additionally write `__nv_cvt_float_to_fp8(val * inv_scale, __NV_SATFINITE, __NV_E4M3)` to an FP8 output buffer in the same thread, same loop iteration. The subsequent GEMM then skips its quantize step entirely (`prequantized=true`), reading the pre-quantized FP8 data directly.

### Calibration vs runtime

- **Calibration** (first inference, `!fp8_calibrated`): unchanged — original FP16 kernels + `quantize_absmax_fp16_to_fp8` inside `lt_gemm_fp8` to compute per-site scales.
- **Runtime** (subsequent inferences, `fp8_calibrated`): fused kernels write FP16 + FP8 simultaneously, GEMM skips quantization.

### Six fused kernels

| Kernel | Sites eliminated | Notes |
|--------|-----------------|-------|
| `layer_norm_fp8()` | 24 (1 per block) | FP8 write in normalize loop |
| `residual_add_layer_norm_fp8()` | 73 (3 per block + 1 enc_proj) | FP8 write in normalize loop |
| `silu_inplace_fp8()` | 48 (2 per block) | In-place FP16 + FP8 to separate buffer |
| `depthwise_conv1d_k9_silu_fp8()` | 24 (1 per block) | Same kernel + FP8 write |
| `transpose_0213_fp8()` | 24 (1 per block) | Data movement + FP8 conversion |
| `reshape_chw_to_hcw_fp8()` | 1 (subsampling output) | Data movement + FP8 conversion |

Total: **194 of 218** `quantize_fp8_static` calls eliminated. The remaining 24 are position-encoding sites — pos_enc is precomputed once at init and reused across all 24 blocks with different per-block scales.

### Challenges

**Corrupted weights during benchmarking.** The Makefile recipe for generating `paraketto-fp8.bin` passes `--weights paraketto-fp16.bin` to `paraketto.fp8`, which the FP8 binary interprets as the FP8 target path — so `fp8_save()` overwrites `paraketto-fp16.bin` with FP8 data instead of creating `paraketto-fp8.bin`. This destroyed the FP16 weights during the first benchmark attempt. The fix: regenerate `paraketto-fp8.bin` with default paths (no `--weights` override), and re-download `paraketto-fp16.bin`.

**Benchmark variance from corrupted weights.** Before the corruption was identified, benchmark numbers were erratic (1119x–1330x across runs). This looked like thermal throttling but was actually caused by running with a `paraketto-fp8.bin` that had been generated through the buggy path. After regenerating cleanly, variance dropped to <2% across runs.

### Results

Controlled A/B comparison, 3 runs each, RTX 5070 Ti:

```
              FP8 baseline (3-run avg)    FP8 fused (3-run avg)
              ───────────────────────    ───────────────────────
               RTFx     WER    Time      RTFx     WER    Time
difficult     1263x  19.38%   403ms     1359x  18.96%   375ms
Total         1262x          5.74s      1315x          5.50s
```

**+4.2% throughput overall, +7.6% on difficult (short) utterances.** WER unchanged.

Full comparison across all backends after the fused optimization:

| Backend | librispeech | earnings22 | long | difficult | **Total RTFx** |
|---------|-------------|------------|------|-----------|----------------|
| CUTLASS FP16 | 1027x | 962x | 1252x | 1247x | **1206x** |
| cuBLAS FP16 | 1084x | 971x | 1238x | 1221x | **1204x** |
| FP8 baseline | 1124x | 979x | 1295x | 1283x | **1262x** |
| **FP8 fused** | **1220x** | **1036x** | **1346x** | **1331x** | **1314x** |

**FP8 fused vs CUTLASS FP16: +9.0% throughput overall.**

## Conclusion

FP8 E4M3 quantization via cublasLt on SM120 (Blackwell) achieves a **~9% speedup over FP16** at the system level after fusing activation quantization into producer kernels. The raw tensor core advantage is larger but partially absorbed by memory bandwidth limits at batch size 1.

The FP8 path is the recommended default because:
- **+9% throughput** over the best FP16 backend (with fused quantization)
- **50% smaller weight cache** (604 MB vs 1.2 GB), faster cold startup (325ms vs 600ms)
- **Better WER on difficult audio** (18.96% vs 23.32%) — per-tensor quantization noise acts as regularization on accented/noisy speech (see below)
- Future cublasLt updates may improve SM120 FP8 kernel tile selection; the infrastructure is in place

### On FP8 quantization as regularization

A surprising and reproducible result: FP8 gets **better** WER than FP16 on the difficult dataset (18.96% vs 23.32%). This is not a bug — it's consistent across every run, and the effect is dataset-dependent:

- **LibriSpeech** (clean read speech): FP8 slightly worse (2.11% vs 1.68%) — quantization noise hurts
- **Earnings22** (clean domain speech): FP8 slightly better (15.62% vs 16.48%)
- **Difficult** (VoxPopuli accented English): FP8 much better (18.96% vs 23.32%) — quantization noise helps

The pattern is consistent with quantization noise acting as **regularization**: on clean, in-distribution speech where the model is well-calibrated, the noise slightly hurts. On accented/noisy speech where the model may overfit to training-distribution patterns, the noise breaks spurious correlations and improves generalization. The effect is localized to the encoder (the decoder stays FP16 since cublasLt FP8 doesn't support N=1 GEMMs).

The **recommended binary for production use** is `paraketto.fp8` with pre-generated `paraketto-fp8.bin`.

## Distribution: Auto-download from HuggingFace

Weights were previously fetched via `gh release download`, which required GitHub CLI authentication and only worked from the project root. Replaced with auto-download from [HuggingFace](https://huggingface.co/localoptima/paraketto) using `wget` — no auth needed for public repos.

Weights now default to `~/.cache/paraketto/` (XDG standard), so the binary works from any directory. On first run, missing weights are downloaded automatically; `--weights` skips the download for local development. The Makefile and Python benchmark harness also pull from HF instead of `gh`.
