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
// LayerNorm + residual: y = x_residual + alpha * LN(x)
//   Fused residual addition with optional scale (alpha=0.5 for FF half-step).
// ---------------------------------------------------------------------------
void layer_norm_residual_fp16(const half* x, const half* x_residual,
                              const half* gamma, const half* beta,
                              half* y, int N, int D, float eps, float alpha,
                              cudaStream_t stream);

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
// SiLU (Swish): y = x * sigmoid(x)
//   Vectorized half2 pointwise kernel.
// ---------------------------------------------------------------------------
void silu_fp16(const half* x, half* y, int n, cudaStream_t stream);

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
// Bias add: y = x + bias (broadcast bias [D] over first dim)
//   x, y: [N, D], bias: [D]
// ---------------------------------------------------------------------------
void bias_add_fp16(const half* x, const half* bias, half* y,
                   int N, int D, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Bias add in-place: x += bias
// ---------------------------------------------------------------------------
void bias_add_inplace_fp16(half* x, const half* bias, int N, int D,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// Add + ReLU (joint network): y = max(a + b, 0)
//   a, b, y: [N]
// ---------------------------------------------------------------------------
void add_relu_fp16(const half* a, const half* b, half* y, int n,
                   cudaStream_t stream);

// ---------------------------------------------------------------------------
// Depthwise conv 1D, kernel=9, stride=1, padding=4 (same)
//   x: [T, C]  (channels-last / row-major)
//   w: [C, 1, 9]
//   b: [C]
//   y: [T, C]
// ---------------------------------------------------------------------------
void depthwise_conv1d_k9_fp16(const half* x, const half* w, const half* b,
                              half* y, int T, int C, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Depthwise conv 1D, kernel=9 + SiLU fused
//   Same as depthwise_conv1d_k9 but applies SiLU(x * sigmoid(x)) to the result.
// ---------------------------------------------------------------------------
void depthwise_conv1d_k9_silu_fp16(const half* x, const half* w, const half* b,
                                    half* y, int T, int C, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Softmax over last dimension
//   x, y: [rows, cols]
// ---------------------------------------------------------------------------
void softmax_fp16(const half* x, half* y, int rows, int cols,
                  cudaStream_t stream);

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
// Embedding gather: y = table[idx]
//   table: [V, D], idx: scalar int, y: [D]
// ---------------------------------------------------------------------------
void embedding_gather_fp16(const half* table, int idx, half* y, int D,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// FP16 -> FP32 cast
// ---------------------------------------------------------------------------
void cast_fp16_to_fp32(const half* x, float* y, int n, cudaStream_t stream);

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
// Relative positional attention score computation
//   Adds pos_bias_u/v to Q, computes content and position attention scores,
//   applies softmax, and produces the weighted sum.
//   This is split into multiple sub-steps called from C++ orchestration.
// ---------------------------------------------------------------------------

// Add pos_bias: q_u = q + pos_bias_u, q_v = q + pos_bias_v
//   q: [heads, T, head_dim], pos_bias: [heads, head_dim]
//   q_u, q_v: [heads, T, head_dim]
void add_pos_bias_fp16(const half* q, const half* pos_bias,
                       half* q_out, int heads, int T, int head_dim,
                       cudaStream_t stream);

// Dual add pos_bias: q_u = q + bias_u, q_v = q + bias_v in one pass
void add_pos_bias_dual_fp16(const half* q,
                             const half* bias_u, const half* bias_v,
                             half* q_u, half* q_v,
                             int heads, int T, int head_dim,
                             cudaStream_t stream);

// Relative position skew: convert [heads, T, 2T-1] position scores
// to [heads, T, T] aligned scores via the Transformer-XL skew trick.
void rel_pos_skew_fp16(const half* pos_scores, half* out,
                       int heads, int T, cudaStream_t stream);

// Scale + add two score matrices: out = (content + position) / sqrt(head_dim)
void scale_add_scores_fp16(const half* content, const half* position,
                           half* out, int heads, int T, float scale,
                           cudaStream_t stream);

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
// ReLU in-place: x = max(x, 0)
// ---------------------------------------------------------------------------
void relu_inplace_fp16(half* x, int n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Reshape [C, H, W] -> [W, C*H] (NCHW to time-major for subsampling output)
//   Reads element (c, h, w) and writes to (w, c*H + h)
// ---------------------------------------------------------------------------
void reshape_chw_to_wch_fp16(const half* in, half* out,
                              int C, int H, int W, cudaStream_t stream);

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
// Split + transpose: [T, 3*heads*head_dim] -> 3x [heads, T, head_dim]
//   Input is [T, 3*D] where D = heads*head_dim.
//   Splits into 3 chunks along last dim and transposes [T, heads, head_dim] -> [heads, T, head_dim].
// ---------------------------------------------------------------------------
void split_transpose_3way_fp16(const half* in,
                                half* out0, half* out1, half* out2,
                                int T, int heads, int head_dim,
                                cudaStream_t stream);
