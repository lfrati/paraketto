// kernels.h — Custom CUDA kernel launch wrappers for Parakeet conformer
//
// All kernels operate on FP16 data with FP32 accumulation where needed.
// Each function launches a CUDA kernel on the given stream.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//   x, y:     [N, D]
//   gamma, beta: [D]
//   One warp per row, FP32 accumulation.
// ---------------------------------------------------------------------------
void layer_norm_fp16(const half* x, const half* gamma, const half* beta,
                     half* y, int N, int D, float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused residual add + LayerNorm:
//   x_out = x + alpha * delta  (update x in-place)
//   ln_out = LN(x_out)         (normalize the updated x)
//   Saves one full T×D read+write vs separate residual_add + layer_norm.
// ---------------------------------------------------------------------------
void residual_add_layer_norm_fp16(half* x, const half* delta, float alpha,
                                   const half* gamma, const half* beta,
                                   half* ln_out, int N, int D, float eps,
                                   cudaStream_t stream);

// ---------------------------------------------------------------------------
// SiLU in-place: x = x * sigmoid(x)
// ---------------------------------------------------------------------------
void silu_inplace_fp16(half* x, int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// GLU (Gated Linear Unit):
//   Input [N, 2D], output [N, D]
//   y = x[:, :D] * sigmoid(x[:, D:])
// ---------------------------------------------------------------------------
void glu_fp16(const half* x, half* y, int N, int D, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Add + ReLU (joint network): y = max(a + b, 0)
//   a, b, y: [N]
// ---------------------------------------------------------------------------
void add_relu_fp16(const half* a, const half* b, half* y, int n,
                   cudaStream_t stream);

// ---------------------------------------------------------------------------
// Dual argmax: token = argmax(logits[:vocab_size]), step = argmax(logits[vocab_size:])
//   logits: [total] FP16, out: [2] int (token, step)
//   One block, 256 threads with warp reduction.
// ---------------------------------------------------------------------------
void dual_argmax_fp16(const half* logits, int* out,
                       int vocab_size, int total, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Depthwise conv 1D, kernel=9 + SiLU fused
//   x: [T, C], w: [C, 1, 9], b: [C], y: [T, C]
// ---------------------------------------------------------------------------
void depthwise_conv1d_k9_silu_fp16(const half* x, const half* w, const half* b,
                                    half* y, int T, int C, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused LSTM cell (single step)
//   gates: [1, 4*H]  (pre-computed: W_ih @ x + W_hh @ h_prev + bias)
//   c_prev: [1, H]
//   h_out, c_out: [1, H]
//   The bias for ONNX LSTM is [1, 8*H] = [b_ih(4*H), b_hh(4*H)]
//   We pre-add both bias halves, so gates input already includes full bias.
// ---------------------------------------------------------------------------
void lstm_cell_fp16(const half* gates, const half* c_prev,
                    half* h_out, half* c_out, int H, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Embed + concat: out[0:D] = table[idx], out[D:2D] = h[0:D]
//   table: [V, D], h: [D], out: [2*D]
//   Single block, D/2 threads (half2 vectorized).
//   Replaces embedding_gather + separate concat for LSTM0 input.
// ---------------------------------------------------------------------------
void embed_concat_fp16(const half* table, int idx, const half* h,
                       half* out, int D, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Concat two D-length vectors: out[0:D] = a, out[D:2D] = b
//   a, b: [D], out: [2*D]
//   Single block, D/2 threads (half2 vectorized).
//   Used for LSTM1 input preparation.
// ---------------------------------------------------------------------------
void concat_vectors_fp16(const half* a, const half* b,
                         half* out, int D, cudaStream_t stream);

// ---------------------------------------------------------------------------
// FP32 -> FP16 cast
// ---------------------------------------------------------------------------
void cast_fp32_to_fp16(const float* x, half* y, int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Transpose 2D: [M, N] -> [N, M]
// ---------------------------------------------------------------------------
void transpose_fp16(const half* x, half* y, int M, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Residual add: y = a + alpha * b
//   a, b, y: [N]
// ---------------------------------------------------------------------------
void residual_add_fp16(const half* a, const half* b, half* y, int n,
                       float alpha, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused skew + scale + softmax for relative position attention
//   Combines rel_pos_skew + scale_add_scores + softmax into one kernel.
//   content_scores: [heads, T, T]
//   pos_scores_raw: [heads, T, 2T-1]
//   output:         [heads, T, T]
// ---------------------------------------------------------------------------
void fused_score_softmax_fp16(const half* content_scores,
                               const half* pos_scores_raw,
                               half* output,
                               int heads, int T, float scale,
                               cudaStream_t stream);

// ---------------------------------------------------------------------------
// Conv2D: general 2D convolution (NCHW format)
//   input:  [C_in, H_in, W_in]
//   weight: [C_out, C_in/groups, kH, kW]
//   bias:   [C_out] (nullable)
//   output: [C_out, H_out, W_out]
//   Supports groups=1 (regular) and groups=C_in (depthwise).
// ---------------------------------------------------------------------------
void conv2d_fp16(const half* input, const half* weight, const half* bias,
                 half* output, int C_in, int H_in, int W_in,
                 int C_out, int kH, int kW, int stride, int pad, int groups,
                 cudaStream_t stream);

// ---------------------------------------------------------------------------
// im2col for 2D convolution (NCHW format)
//   input:  [C_in, H_in, W_in]
//   col:    [C_in*kH*kW, H_out*W_out]  (row-major)
//   Extracts sliding window patches for use with GEMM-based convolution.
// ---------------------------------------------------------------------------
void im2col_2d_fp16(const half* input, half* col,
                    int C_in, int H_in, int W_in,
                    int kH, int kW, int stride, int pad,
                    int H_out, int W_out, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Per-channel bias + ReLU for NCHW data (in-place)
//   x[c, s] = max(x[c, s] + bias[c], 0) for s in [0, spatial)
// ---------------------------------------------------------------------------
void bias_relu_nchw_fp16(half* x, const half* bias, int C, int spatial,
                          cudaStream_t stream);

// ---------------------------------------------------------------------------
// Reshape [C, H, W] -> [H, C*W] (permute(1,0,2) + flatten for subsampling)
//   Reads element (c, h, w) and writes to (h, c*W + w)
// ---------------------------------------------------------------------------
void reshape_chw_to_hcw_fp16(const half* in, half* out,
                              int C, int H, int W, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Generate sinusoidal position encoding on GPU
//   output: [(2*T-1), d_model]
//   positions: T-1, T-2, ..., 0, -1, ..., -(T-1)
//   pe[p][2i]   = sin(pos / 10000^(2i/d))
//   pe[p][2i+1] = cos(pos / 10000^(2i/d))
// ---------------------------------------------------------------------------
void generate_pos_encoding_gpu(half* output, int T, int d_model,
                                cudaStream_t stream);

// ---------------------------------------------------------------------------
// Transpose [A, B, C] -> [B, A, C]  (swap first two dims)
//   Used for attention head reshaping: [T, heads, head_dim] <-> [heads, T, head_dim]
// ---------------------------------------------------------------------------
void transpose_0213_fp16(const half* in, half* out,
                          int A, int B, int C, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused split + transpose + pos_bias: [T, 3*D] -> q_u, q_v (with bias), K, V
//   Reads QKV interleaved data once, writes:
//     q_u[H,T,d] = Q[h,t,d] + bias_u[h,d]
//     q_v[H,T,d] = Q[h,t,d] + bias_v[h,d]
//     K[H,T,d], V[H,T,d]  (unchanged)
//   Eliminates separate split_transpose_3way + add_pos_bias_dual.
// ---------------------------------------------------------------------------
void split_transpose_qkv_bias_fp16(const half* in,
                                    const half* bias_u,
                                    const half* bias_v,
                                    half* q_u, half* q_v,
                                    half* k, half* v,
                                    int T, int heads, int head_dim,
                                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Mel filterbank entry for fused GPU mel pipeline
// ---------------------------------------------------------------------------
struct MelFBEntry {
    uint16_t freq;
    uint16_t mel;
    float weight;
};

// ---------------------------------------------------------------------------
// Initialize mel filterbank data in GPU constant memory.
// Must be called once before fft512_mel_log.
//   entries: [count] MelFBEntry structs (504 for Parakeet)
// ---------------------------------------------------------------------------
void mel_init_filterbank(const MelFBEntry* entries, int count);

// ---------------------------------------------------------------------------
// Fused FFT + mel filterbank + log
//   frames:  [n_frames, 512]  real windowed audio frames (float32)
//   mel_out: [n_frames, 128]  log-mel spectrogram (float32)
//   One block per frame, 256 threads. Fuses FFT -> power -> sparse mel -> log.
// ---------------------------------------------------------------------------
void fft512_mel_log(const float* frames, float* mel_out, int n_frames,
                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Per-channel mel normalize + transpose
//   mel_in:  [n_frames, 128]  log-mel from fft512_mel_log (float32)
//   mel_out: [128, n_valid]   normalized, transposed (float32)
//   One block per mel channel (128 blocks). Computes mean/variance over
//   n_valid frames, normalizes with Bessel correction: (x - mean) / (std + eps).
// ---------------------------------------------------------------------------
void mel_normalize(const float* mel_in, float* mel_out,
                   int n_frames, int n_valid, cudaStream_t stream);
