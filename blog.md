# Parakeet Native: Building a Single-Executable ASR Engine

## Goal

Replace the current Python + onnx-asr + TensorRT stack with a single compiled executable that runs NVIDIA's Parakeet TDT 0.6B V2 faster than the Python baseline. No Python, no pip, no virtualenv — just one binary and the model weights.

## Current Baseline

Our existing implementation (`src/parakeet_trt.py`) uses Python's `onnx-asr` library with TensorRT acceleration via ONNX Runtime. Benchmark results on our test hardware:

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

The most architecturally interesting project. A from-scratch C++ implementation that does NOT use ONNX Runtime — instead it uses [Axiom](https://github.com/Frikallo/axiom), a custom lightweight tensor library with Metal and CUDA backends.

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

The most mature and battle-tested option. 10.5k stars, 1.2k forks, 1687 commits. C++ core with bindings for 12 languages.

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

The pragmatic choice. ONNX Runtime's C++ API has essentially identical performance to the Python wrapper because the actual GPU compute happens in the same C++ backend. When configured with the TensorRT execution provider, it should match our current ~850x RTFx.

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

**Key insight:** Nobody has yet built a production-ready, single-binary, GPU-accelerated Parakeet executable for Linux with CUDA. parakeet.cpp (Frikallo) is the closest architecturally but targets Apple Metal. sherpa-onnx is the most mature but depends on ONNX Runtime shared libs.

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
C++ (parakeet.cpp):
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

Long-audio mel time: 42.9ms → 28.4ms (1.5x faster). Total RTFx: 918x → **1049x**.

### Final Results

```
C++ (parakeet.cpp):
  librispeech: WER=1.81% RTFx=827x (40 utts, 276s audio)
  earnings22:  WER=16.48% RTFx=796x (40 utts, 253s audio)
  long-audio:  RTFx=1049x (92s audio, 87ms)

Python (parakeet_trt.py via onnx-asr):
  librispeech: WER=1.81% RTFx=529x (40 utts, 276s audio)
  earnings22:  WER=16.48% RTFx=603x (40 utts, 253s audio)
  long-audio:  RTFx=874x (92s audio, 105ms)
```

WER matches Python exactly on both datasets. The C++ binary is **20% faster** on long-audio and **~50% faster** on short clips (no Python/ORT overhead per utterance). Non-16kHz audio is rejected at load time (the model expects 16kHz input).

### Architecture

```
src/parakeet.cpp     # ~550 lines, single file
engines/
  encoder.engine     # 1.2 GB, FP16
  decoder_joint.engine  # 18 MB, FP16
model/
  vocab.txt          # 1025 BPE tokens
  hann_window.bin    # 512 floats
  mel_filterbank.bin # 257x128 floats
Makefile
```

Pipeline: WAV → CPU mel spectrogram → GPU TRT encoder → GPU TRT decoder (greedy TDT loop) → BPE detokenize → text

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

**Early `cudaFree(0)`.** Forces CUDA driver initialization (~106ms warm, ~300ms cold) before engine loading. This didn't reduce total time — CUDA init was already happening inside the first `deserializeCudaEngine` call. But it made the cost visible and separable, and prevents surprises if the init order changes.

**Pre-create cuFFT plan.** Called `mel.ensure_fft(1000)` during init to build the cuFFT plan before the first transcription. Saved ~1ms from warmup. Negligible but zero-cost.

**WAV readahead with `posix_fadvise`.** Hints the kernel to pre-cache the first WAV file while engines load. No measurable effect on test data (WAV files are ~110KB), but zero-cost insurance for large audio files.

**Reuse warmup WAV data.** The first file was being read and transcribed twice — once for warmup (discarded), once for real output. Now `std::move`s the warmup WAV data into the processing loop, avoiding the redundant read and allocation.

#### What didn't help

**`-O3 -march=native -flto` compiler flags.** No measurable startup improvement — startup is I/O and TRT deserialization, not CPU compute. Kept the flags anyway since they benefit inference runtime (mel spectrogram vectorization, decoder argmax).

**Pooling GPU buffer allocations (skipped).** The 11 separate `cudaMalloc` calls for decoder state buffers already took <0.5ms combined. Pooling into a single allocation would add complexity for no measurable gain.

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
