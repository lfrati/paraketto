// kernels.cu — Custom CUDA kernels for Parakeet conformer
//
// All kernels use FP16 I/O with FP32 accumulation where needed.
// Block/grid sizes tuned for typical conformer dimensions (D=1024, H=640).

#include "kernels.h"
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

__global__ void layer_norm_kernel(const half* __restrict__ x,
                                  const half* __restrict__ gamma,
                                  const half* __restrict__ beta,
                                  half* __restrict__ y,
                                  int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const half* xr = x + row * D;
    half* yr = y + row * D;

    // Compute mean and variance with warp-level reduction
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        sum += v;
        sum2 += v * v;
    }

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum  += __shfl_xor_sync(0xffffffff, sum, mask);
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
    }

    // Block reduction via shared memory (for blockDim > 32)
    __shared__ float s_sum[32], s_sum2[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (lane == 0) { s_sum[warp_id] = sum; s_sum2[warp_id] = sum2; }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        sum  = (lane < nwarps) ? s_sum[lane] : 0.0f;
        sum2 = (lane < nwarps) ? s_sum2[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum  += __shfl_xor_sync(0xffffffff, sum, mask);
            sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
        }
    }

    // Broadcast mean and inv_std
    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        s_mean = sum / D;
        float var = sum2 / D - s_mean * s_mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        yr[i] = __float2half((v - mean) * inv_std * g + b);
    }
}

void layer_norm_fp16(const half* x, const half* gamma, const half* beta,
                     half* y, int N, int D, float eps, cudaStream_t stream) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    layer_norm_kernel<<<N, threads, 0, stream>>>(x, gamma, beta, y, N, D, eps);
}

// ---------------------------------------------------------------------------
// Fused residual add + LayerNorm
//   x_out[row] = x[row] + alpha * delta[row]   (writes back to x)
//   ln_out[row] = LN(x_out[row])               (normalized output)
// ---------------------------------------------------------------------------

__global__ void residual_add_layer_norm_kernel(half* __restrict__ x,
                                                const half* __restrict__ delta,
                                                float alpha,
                                                const half* __restrict__ gamma,
                                                const half* __restrict__ beta,
                                                half* __restrict__ ln_out,
                                                int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    half* xr = x + row * D;
    const half* dr = delta + row * D;
    half* yr = ln_out + row * D;

    // Pass 1: update x and compute mean/var
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]) + alpha * __half2float(dr[i]);
        xr[i] = __float2half(v);  // write back updated x
        sum += v;
        sum2 += v * v;
    }

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum  += __shfl_xor_sync(0xffffffff, sum, mask);
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
    }

    __shared__ float s_sum[32], s_sum2[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) { s_sum[warp_id] = sum; s_sum2[warp_id] = sum2; }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        sum  = (lane < nwarps) ? s_sum[lane] : 0.0f;
        sum2 = (lane < nwarps) ? s_sum2[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum  += __shfl_xor_sync(0xffffffff, sum, mask);
            sum2 += __shfl_xor_sync(0xffffffff, sum2, mask);
        }
    }

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        s_mean = sum / D;
        float var = sum2 / D - s_mean * s_mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Pass 2: normalize (read from updated x)
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        yr[i] = __float2half((v - mean) * inv_std * g + b);
    }
}

void residual_add_layer_norm_fp16(half* x, const half* delta, float alpha,
                                   const half* gamma, const half* beta,
                                   half* ln_out, int N, int D, float eps,
                                   cudaStream_t stream) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    residual_add_layer_norm_kernel<<<N, threads, 0, stream>>>(
        x, delta, alpha, gamma, beta, ln_out, N, D, eps);
}

// ---------------------------------------------------------------------------
// SiLU (in-place)
// ---------------------------------------------------------------------------

__global__ void silu_inplace_kernel(half* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        x[i] = __float2half(v / (1.0f + expf(-v)));
    }
}

void silu_inplace_fp16(half* x, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_inplace_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

// ---------------------------------------------------------------------------
// GLU
// ---------------------------------------------------------------------------

__global__ void glu_kernel(const half* __restrict__ x, half* __restrict__ y,
                           int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D;
    if (idx < total) {
        int row = idx / D;
        int col = idx % D;
        float a = __half2float(x[row * 2 * D + col]);
        float b = __half2float(x[row * 2 * D + D + col]);
        y[idx] = __float2half(a * (1.0f / (1.0f + expf(-b))));
    }
}

void glu_fp16(const half* x, half* y, int N, int D, cudaStream_t stream) {
    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    glu_kernel<<<blocks, threads, 0, stream>>>(x, y, N, D);
}

// ---------------------------------------------------------------------------
// Add + ReLU
// ---------------------------------------------------------------------------

__global__ void add_relu_kernel(const half* __restrict__ a,
                                const half* __restrict__ b,
                                half* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(a[i]) + __half2float(b[i]);
        y[i] = __float2half(v > 0.0f ? v : 0.0f);
    }
}

void add_relu_fp16(const half* a, const half* b, half* y, int n,
                   cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_relu_kernel<<<blocks, threads, 0, stream>>>(a, b, y, n);
}

// ---------------------------------------------------------------------------
// Dual argmax: finds argmax over [0, vocab_size) and [vocab_size, total)
//   logits: [total] FP16 values
//   out: [2] int — out[0] = token (argmax over vocab), out[1] = step (argmax over durations)
//   One block, 256 threads, warp-level reduction.
// ---------------------------------------------------------------------------

__global__ void dual_argmax_kernel(const half* __restrict__ logits,
                                    int* __restrict__ out,
                                    int vocab_size, int total) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // --- Token argmax: [0, vocab_size) ---
    float best_val = -FLT_MAX;
    int best_idx = 0;
    for (int i = tid; i < vocab_size; i += nthreads) {
        float v = __half2float(logits[i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, mask);
        int other_idx = __shfl_xor_sync(0xffffffff, best_idx, mask);
        if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }

    __shared__ float s_val[32];
    __shared__ int s_idx[32];
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) { s_val[warp_id] = best_val; s_idx[warp_id] = best_idx; }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = nthreads / 32;
        best_val = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
        best_idx = (lane < nwarps) ? s_idx[lane] : 0;
        for (int mask = 16; mask > 0; mask >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, mask);
            int other_idx = __shfl_xor_sync(0xffffffff, best_idx, mask);
            if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
        }
        if (lane == 0) out[0] = best_idx;
    }
    __syncthreads();

    // --- Step argmax: [vocab_size, total) ---
    int dur_size = total - vocab_size;
    best_val = -FLT_MAX;
    best_idx = 0;
    for (int i = tid; i < dur_size; i += nthreads) {
        float v = __half2float(logits[vocab_size + i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }

    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, best_val, mask);
        int other_idx = __shfl_xor_sync(0xffffffff, best_idx, mask);
        if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }

    if (lane == 0) { s_val[warp_id] = best_val; s_idx[warp_id] = best_idx; }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = nthreads / 32;
        best_val = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
        best_idx = (lane < nwarps) ? s_idx[lane] : 0;
        for (int mask = 16; mask > 0; mask >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, mask);
            int other_idx = __shfl_xor_sync(0xffffffff, best_idx, mask);
            if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
        }
        if (lane == 0) out[1] = best_idx;
    }
}

void dual_argmax_fp16(const half* logits, int* out,
                       int vocab_size, int total, cudaStream_t stream) {
    dual_argmax_kernel<<<1, 256, 0, stream>>>(logits, out, vocab_size, total);
}

// ---------------------------------------------------------------------------
// Depthwise conv 1D, kernel=9 + SiLU fused
// ---------------------------------------------------------------------------

__global__ void depthwise_conv1d_k9_silu_kernel(const half* __restrict__ x,
                                                 const half* __restrict__ w,
                                                 const half* __restrict__ b,
                                                 half* __restrict__ y,
                                                 int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * C;
    if (idx < total) {
        int t = idx / C;
        int c = idx % C;

        float sum = 0.0f;
        const half* wc = w + c * 9;

        #pragma unroll
        for (int k = 0; k < 9; k++) {
            int ti = t + k - 4;  // padding=4
            if (ti >= 0 && ti < T) {
                sum += __half2float(x[ti * C + c]) * __half2float(wc[k]);
            }
        }

        if (b) sum += __half2float(b[c]);
        // SiLU: x * sigmoid(x)
        y[idx] = __float2half(sum / (1.0f + expf(-sum)));
    }
}

void depthwise_conv1d_k9_silu_fp16(const half* x, const half* w, const half* b,
                                    half* y, int T, int C, cudaStream_t stream) {
    int total = T * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    depthwise_conv1d_k9_silu_kernel<<<blocks, threads, 0, stream>>>(x, w, b, y, T, C);
}

// ---------------------------------------------------------------------------
// LSTM cell
// ---------------------------------------------------------------------------

__global__ void lstm_cell_kernel(const half* __restrict__ gates,
                                 const half* __restrict__ c_prev,
                                 half* __restrict__ h_out,
                                 half* __restrict__ c_out,
                                 int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) {
        // gates layout: [i, o, f, g] each of size H (ONNX LSTM gate order: iofc)
        float gi = __half2float(gates[i]);          // input gate
        float go = __half2float(gates[H + i]);       // output gate
        float gf = __half2float(gates[2 * H + i]);   // forget gate
        float gg = __half2float(gates[3 * H + i]);   // cell gate

        float i_gate = 1.0f / (1.0f + expf(-gi));
        float o_gate = 1.0f / (1.0f + expf(-go));
        float f_gate = 1.0f / (1.0f + expf(-gf));
        float c_gate = tanhf(gg);

        float c = f_gate * __half2float(c_prev[i]) + i_gate * c_gate;
        float h = o_gate * tanhf(c);

        c_out[i] = __float2half(c);
        h_out[i] = __float2half(h);
    }
}

void lstm_cell_fp16(const half* gates, const half* c_prev,
                    half* h_out, half* c_out, int H, cudaStream_t stream) {
    int threads = 256;
    int blocks = (H + threads - 1) / threads;
    lstm_cell_kernel<<<blocks, threads, 0, stream>>>(gates, c_prev, h_out, c_out, H);
}

// ---------------------------------------------------------------------------
// Embed + concat: out[0:D] = table[idx], out[D:2D] = h[0:D]
// Single block, half2 vectorized.
// ---------------------------------------------------------------------------

__global__ void embed_concat_kernel(const half* __restrict__ table,
                                     int idx, const half* __restrict__ h,
                                     half* __restrict__ out, int D) {
    int tid = threadIdx.x;
    int D2 = D / 2;  // half2 count per vector
    const half2* table2 = (const half2*)(table + idx * D);
    const half2* h2 = (const half2*)h;
    half2* out2 = (half2*)out;
    // Each thread copies one half2 from embed and one from h
    for (int i = tid; i < D2; i += blockDim.x) {
        out2[i] = table2[i];        // out[0:D]
        out2[D2 + i] = h2[i];       // out[D:2D]
    }
}

void embed_concat_fp16(const half* table, int idx, const half* h,
                       half* out, int D, cudaStream_t stream) {
    embed_concat_kernel<<<1, 256, 0, stream>>>(table, idx, h, out, D);
}

// ---------------------------------------------------------------------------
// Concat two D-length vectors: out[0:D] = a, out[D:2D] = b
// Single block, half2 vectorized.
// ---------------------------------------------------------------------------

__global__ void concat_vectors_kernel(const half* __restrict__ a,
                                       const half* __restrict__ b,
                                       half* __restrict__ out, int D) {
    int tid = threadIdx.x;
    int D2 = D / 2;
    const half2* a2 = (const half2*)a;
    const half2* b2 = (const half2*)b;
    half2* out2 = (half2*)out;
    for (int i = tid; i < D2; i += blockDim.x) {
        out2[i] = a2[i];
        out2[D2 + i] = b2[i];
    }
}

void concat_vectors_fp16(const half* a, const half* b,
                         half* out, int D, cudaStream_t stream) {
    concat_vectors_kernel<<<1, 256, 0, stream>>>(a, b, out, D);
}

// ---------------------------------------------------------------------------
// FP16 <-> FP32 casts
// ---------------------------------------------------------------------------

__global__ void cast_fp32_to_fp16_kernel(const float* __restrict__ x,
                                         half* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

void cast_fp32_to_fp16(const float* x, half* y, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cast_fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}

// ---------------------------------------------------------------------------
// Transpose 2D: [M, N] -> [N, M] using tiled approach for coalescing
// ---------------------------------------------------------------------------

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_kernel(const half* __restrict__ x, half* __restrict__ y,
                                 int M, int N) {
    __shared__ half tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int xIdx = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIdx = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input [M, N]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (yIdx + j < M && xIdx < N)
            tile[threadIdx.y + j][threadIdx.x] = x[(yIdx + j) * N + xIdx];
    }
    __syncthreads();

    // Write transposed tile to output [N, M]
    int outX = blockIdx.y * TILE_DIM + threadIdx.x;
    int outY = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (outY + j < N && outX < M)
            y[(outY + j) * M + outX] = tile[threadIdx.x][threadIdx.y + j];
    }
}

void transpose_fp16(const half* x, half* y, int M, int N, cudaStream_t stream) {
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    transpose_kernel<<<blocks, threads, 0, stream>>>(x, y, M, N);
}

// ---------------------------------------------------------------------------
// Residual add: y = a + alpha * b
// ---------------------------------------------------------------------------

__global__ void residual_add_kernel(const half* __restrict__ a,
                                    const half* __restrict__ b,
                                    half* __restrict__ y, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half(__half2float(a[i]) + alpha * __half2float(b[i]));
    }
}

void residual_add_fp16(const half* a, const half* b, half* y, int n,
                       float alpha, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, y, n, alpha);
}

// ---------------------------------------------------------------------------
// Fused skew + scale + softmax
// One thread block per row (H*T total rows), each row of length T.
// Computes: score[j] = (content[j] + pos_raw[j + T-1-t]) * scale
// then softmax over the row.
// ---------------------------------------------------------------------------

__global__ void fused_score_softmax_kernel(const half* __restrict__ content,
                                            const half* __restrict__ pos_raw,
                                            half* __restrict__ output,
                                            int T, float scale) {
    int row = blockIdx.x;  // row in [0, heads*T)
    int h = row / T;
    int t = row % T;
    int W = 2 * T - 1;

    const half* content_row = content + row * T;
    const half* pos_row = pos_raw + h * T * W + t * W;
    half* out_row = output + row * T;

    // Pass 1: compute scaled scores and find max
    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        int src_col = j + T - 1 - t;  // skew index
        float c = __half2float(content_row[j]);
        float p = __half2float(pos_row[src_col]);
        float s = (c + p) * scale;
        // Store temporarily in output to avoid recomputation
        out_row[j] = __float2half(s);
        if (s > max_val) max_val = s;
    }

    // Warp reduction for max
    for (int mask = 16; mask > 0; mask >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));

    __shared__ float s_max[32];
    int warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) s_max[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        max_val = (lane < nwarps) ? s_max[lane] : -FLT_MAX;
        for (int mask = 16; mask > 0; mask >>= 1)
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask));
    }
    __shared__ float s_max_final;
    if (threadIdx.x == 0) s_max_final = max_val;
    __syncthreads();
    max_val = s_max_final;

    // Pass 2: exp and sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        sum += expf(__half2float(out_row[j]) - max_val);

    for (int mask = 16; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        sum = (lane < nwarps) ? s_sum[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    __shared__ float s_inv_sum;
    if (threadIdx.x == 0) s_inv_sum = 1.0f / sum;
    __syncthreads();
    float inv_sum = s_inv_sum;

    // Pass 3: normalize
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        out_row[j] = __float2half(expf(__half2float(out_row[j]) - max_val) * inv_sum);
}

void fused_score_softmax_fp16(const half* content_scores,
                               const half* pos_scores_raw,
                               half* output,
                               int heads, int T, float scale,
                               cudaStream_t stream) {
    int rows = heads * T;
    int threads = (T <= 1024) ? ((T + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    fused_score_softmax_kernel<<<rows, threads, 0, stream>>>(
        content_scores, pos_scores_raw, output, T, scale);
}

// ---------------------------------------------------------------------------
// Conv2D (general, NCHW format)
// ---------------------------------------------------------------------------

__global__ void conv2d_kernel(const half* __restrict__ input,
                              const half* __restrict__ weight,
                              const half* __restrict__ bias,
                              half* __restrict__ output,
                              int C_in, int H_in, int W_in,
                              int C_out, int H_out, int W_out,
                              int kH, int kW, int stride, int pad,
                              int groups, int c_per_group) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * H_out * W_out;
    if (idx >= total) return;

    int oc = idx / (H_out * W_out);
    int rem = idx % (H_out * W_out);
    int oh = rem / W_out;
    int ow = rem % W_out;

    int group = oc / (C_out / groups);
    int ic_start = group * c_per_group;

    float sum = 0.0f;
    for (int ic = 0; ic < c_per_group; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int ih = oh * stride + kh - pad;
                int iw = ow * stride + kw - pad;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    float x = __half2float(input[(ic_start + ic) * H_in * W_in + ih * W_in + iw]);
                    float w = __half2float(weight[oc * c_per_group * kH * kW + ic * kH * kW + kh * kW + kw]);
                    sum += x * w;
                }
            }
        }
    }
    if (bias) sum += __half2float(bias[oc]);
    output[idx] = __float2half(sum);
}

void conv2d_fp16(const half* input, const half* weight, const half* bias,
                 half* output, int C_in, int H_in, int W_in,
                 int C_out, int kH, int kW, int stride, int pad, int groups,
                 cudaStream_t stream) {
    int H_out = (H_in + 2 * pad - kH) / stride + 1;
    int W_out = (W_in + 2 * pad - kW) / stride + 1;
    int c_per_group = C_in / groups;
    int total = C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, C_in, H_in, W_in,
        C_out, H_out, W_out, kH, kW, stride, pad, groups, c_per_group);
}

// ---------------------------------------------------------------------------
// im2col for 2D convolution (NCHW format)
// Extracts patches: col[c*kH*kW + kh*kW + kw, oh*W_out + ow] = input[c, oh*s+kh-p, ow*s+kw-p]
// ---------------------------------------------------------------------------

__global__ void im2col_2d_kernel(const half* __restrict__ input, half* __restrict__ col,
                                  int C_in, int H_in, int W_in,
                                  int kH, int kW, int stride, int pad,
                                  int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_rows = C_in * kH * kW;
    int col_cols = H_out * W_out;
    int total = col_rows * col_cols;
    if (idx >= total) return;

    int spatial = idx % col_cols;
    int patch = idx / col_cols;

    int oh = spatial / W_out;
    int ow = spatial % W_out;

    int ic = patch / (kH * kW);
    int kk = patch % (kH * kW);
    int kh = kk / kW;
    int kw = kk % kW;

    int ih = oh * stride + kh - pad;
    int iw = ow * stride + kw - pad;

    half val = __float2half(0.0f);
    if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)
        val = input[ic * H_in * W_in + ih * W_in + iw];

    col[idx] = val;
}

void im2col_2d_fp16(const half* input, half* col,
                    int C_in, int H_in, int W_in,
                    int kH, int kW, int stride, int pad,
                    int H_out, int W_out, cudaStream_t stream) {
    int total = C_in * kH * kW * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    im2col_2d_kernel<<<blocks, threads, 0, stream>>>(
        input, col, C_in, H_in, W_in, kH, kW, stride, pad, H_out, W_out);
}

// ---------------------------------------------------------------------------
// Per-channel bias + ReLU for NCHW data (in-place)
// x is [C, spatial] row-major. Adds bias[c] to channel c, applies ReLU.
// ---------------------------------------------------------------------------

__global__ void bias_relu_nchw_kernel(half* __restrict__ x, const half* __restrict__ bias,
                                       int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    int c = idx / spatial;
    float val = __half2float(x[idx]) + __half2float(bias[c]);
    x[idx] = __float2half(fmaxf(val, 0.0f));
}

void bias_relu_nchw_fp16(half* x, const half* bias, int C, int spatial,
                          cudaStream_t stream) {
    int total = C * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_relu_nchw_kernel<<<blocks, threads, 0, stream>>>(x, bias, C, spatial);
}

// ---------------------------------------------------------------------------
// Reshape [C, H, W] -> [H, C*W]  (permute(1,0,2) then flatten last two)
//   Reads element (c, h, w) and writes to (h, c*W + w)
// ---------------------------------------------------------------------------

__global__ void reshape_chw_to_hcw_kernel(const half* __restrict__ in,
                                           half* __restrict__ out,
                                           int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;
    // Output layout: [H, C*W] — idx = h * (C*W) + c * W + w
    int h = idx / (C * W);
    int cw = idx % (C * W);
    int c = cw / W;
    int w = cw % W;
    out[idx] = in[c * H * W + h * W + w];
}

void reshape_chw_to_hcw_fp16(const half* in, half* out,
                              int C, int H, int W, cudaStream_t stream) {
    int total = C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_chw_to_hcw_kernel<<<blocks, threads, 0, stream>>>(in, out, C, H, W);
}

// ---------------------------------------------------------------------------
// Generate sinusoidal position encoding on GPU
// ---------------------------------------------------------------------------

__global__ void generate_pos_encoding_kernel(half* __restrict__ output,
                                              int T, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = 2 * T - 1;
    int half_d = d_model / 2;
    int total = len * half_d;
    if (idx < total) {
        int p = idx / half_d;
        int i = idx % half_d;
        float pos = (float)(T - 1 - p);
        float div_term = expf(-(float)(2 * i) * logf(10000.0f) / d_model);
        float angle = pos * div_term;
        output[p * d_model + 2 * i]     = __float2half(sinf(angle));
        output[p * d_model + 2 * i + 1] = __float2half(cosf(angle));
    }
}

void generate_pos_encoding_gpu(half* output, int T, int d_model,
                                cudaStream_t stream) {
    int len = 2 * T - 1;
    int total = len * (d_model / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    generate_pos_encoding_kernel<<<blocks, threads, 0, stream>>>(
        output, T, d_model);
}

// ---------------------------------------------------------------------------
// Transpose [A, B, C] -> [B, A, C]
// ---------------------------------------------------------------------------

__global__ void transpose_0213_kernel(const half* __restrict__ in,
                                       half* __restrict__ out,
                                       int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx < total) {
        int a = idx / (B * C);
        int rem = idx % (B * C);
        int b = rem / C;
        int c = rem % C;
        out[b * A * C + a * C + c] = in[idx];
    }
}

void transpose_0213_fp16(const half* in, half* out,
                          int A, int B, int C, cudaStream_t stream) {
    int total = A * B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_0213_kernel<<<blocks, threads, 0, stream>>>(in, out, A, B, C);
}

// ---------------------------------------------------------------------------
// Fused split + transpose + pos_bias
// Input: [T, 3*D], bias_u/bias_v: [heads, head_dim]
// Output: q_u, q_v (with bias), K, V in [heads, T, head_dim]
// ---------------------------------------------------------------------------

__global__ void split_transpose_qkv_bias_kernel(const half* __restrict__ in,
                                                  const half* __restrict__ bias_u,
                                                  const half* __restrict__ bias_v,
                                                  half* __restrict__ q_u,
                                                  half* __restrict__ q_v,
                                                  half* __restrict__ k,
                                                  half* __restrict__ v,
                                                  int T, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int D = heads * head_dim;
    int total = T * D;
    if (idx < total) {
        int t = idx / D;
        int rem = idx % D;
        int h = rem / head_dim;
        int d = rem % head_dim;
        int in_row = t * 3 * D;
        int out_idx = h * T * head_dim + t * head_dim + d;

        float q_val = __half2float(in[in_row + rem]);
        float bu = __half2float(bias_u[h * head_dim + d]);
        float bv = __half2float(bias_v[h * head_dim + d]);

        q_u[out_idx] = __float2half(q_val + bu);
        q_v[out_idx] = __float2half(q_val + bv);
        k[out_idx] = in[in_row + D + rem];
        v[out_idx] = in[in_row + 2 * D + rem];
    }
}

void split_transpose_qkv_bias_fp16(const half* in,
                                    const half* bias_u,
                                    const half* bias_v,
                                    half* q_u, half* q_v,
                                    half* k, half* v,
                                    int T, int heads, int head_dim,
                                    cudaStream_t stream) {
    int total = T * heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    split_transpose_qkv_bias_kernel<<<blocks, threads, 0, stream>>>(
        in, bias_u, bias_v, q_u, q_v, k, v, T, heads, head_dim);
}

// ---------------------------------------------------------------------------
// Fused FFT + mel filterbank + log
// One block per frame, 256 threads. Fuses FFT → power → sparse mel → log
// in shared memory, eliminating GPU↔CPU roundtrips.
// ---------------------------------------------------------------------------

static constexpr int N_MEL_FB_ENTRIES = 504;

__constant__ MelFBEntry c_mel_fb[N_MEL_FB_ENTRIES];

void mel_init_filterbank(const MelFBEntry* entries, int count) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_mel_fb, entries, count * sizeof(MelFBEntry)));
}

__global__ void fft512_mel_log_kernel(const float* __restrict__ frames,
                                       float* __restrict__ mel_out,
                                       int n_frames) {
    int frame = blockIdx.x;
    if (frame >= n_frames) return;

    __shared__ float sr[512], si[512];
    int tid = threadIdx.x;  // 0..255

    // 1. Bit-reversal load (identical to fft512_power_kernel)
    const float* in = frames + frame * 512;
    int i0 = tid, i1 = tid + 256;
    int br0 = __brev(i0) >> 23;  // 9-bit reversal
    int br1 = __brev(i1) >> 23;
    sr[br0] = in[i0];  si[br0] = 0.0f;
    sr[br1] = in[i1];  si[br1] = 0.0f;
    __syncthreads();

    // 2. 9 butterfly stages (identical to fft512_power_kernel)
    for (int s = 0; s < 9; s++) {
        int size = 1 << (s + 1);
        int half_size = size >> 1;
        int group = tid / half_size;
        int k = tid % half_size;
        int a = group * size + k;
        int b = a + half_size;

        float angle = -6.283185307179586f * k / size;
        float wr, wi;
        __sincosf(angle, &wi, &wr);

        float tr = wr * sr[b] - wi * si[b];
        float ti = wr * si[b] + wi * sr[b];
        float ar = sr[a], ai = si[a];

        sr[a] = ar + tr;  si[a] = ai + ti;
        sr[b] = ar - tr;  si[b] = ai - ti;
        __syncthreads();
    }

    // 3. Power spectrum → sr[0..256]
    float power_tid = sr[tid] * sr[tid] + si[tid] * si[tid];
    float power_256 = 0.0f;
    if (tid == 0) power_256 = sr[256] * sr[256] + si[256] * si[256];
    __syncthreads();

    sr[tid] = power_tid;
    if (tid == 0) sr[256] = power_256;
    __syncthreads();

    // 4. Mel filterbank: scatter with atomicAdd in shared memory
    // Reuse si[0..127] as mel accumulators
    if (tid < 128) si[tid] = 0.0f;
    __syncthreads();

    // Each thread handles ~2 filterbank entries (504 / 256 ≈ 2)
    int fb_start = tid * N_MEL_FB_ENTRIES / 256;
    int fb_end = (tid + 1) * N_MEL_FB_ENTRIES / 256;
    for (int i = fb_start; i < fb_end; i++) {
        MelFBEntry e = c_mel_fb[i];
        atomicAdd(&si[e.mel], sr[e.freq] * e.weight);
    }
    __syncthreads();

    // 5. Log + write output
    if (tid < 128) {
        mel_out[frame * 128 + tid] = logf(si[tid] + 5.9604645e-08f);
    }
}

void fft512_mel_log(const float* frames, float* mel_out, int n_frames,
                    cudaStream_t stream) {
    fft512_mel_log_kernel<<<n_frames, 256, 0, stream>>>(frames, mel_out, n_frames);
}

// ---------------------------------------------------------------------------
// Per-channel mel normalize + transpose
// One block per mel channel (128 blocks). Two-pass: compute mean/var, then
// normalize and write transposed [128, n_valid].
// ---------------------------------------------------------------------------

__global__ void mel_normalize_kernel(const float* __restrict__ mel_in,
                                      float* __restrict__ mel_out,
                                      int n_frames, int n_valid) {
    int ch = blockIdx.x;  // 0..127
    if (ch >= 128) return;

    int T = n_valid;

    // --- Pass 1: compute mean ---
    float sum = 0.0f;
    for (int f = threadIdx.x; f < T; f += blockDim.x)
        sum += mel_in[f * 128 + ch];

    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    // Block reduction
    __shared__ float s_buf[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) s_buf[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        sum = (lane < nwarps) ? s_buf[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) s_mean = sum / T;
    __syncthreads();
    float mean = s_mean;

    // --- Pass 2: compute variance ---
    float sq = 0.0f;
    for (int f = threadIdx.x; f < T; f += blockDim.x) {
        float d = mel_in[f * 128 + ch] - mean;
        sq += d * d;
    }

    for (int mask = 16; mask > 0; mask >>= 1)
        sq += __shfl_xor_sync(0xffffffff, sq, mask);

    if (lane == 0) s_buf[warp_id] = sq;
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        sq = (lane < nwarps) ? s_buf[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            sq += __shfl_xor_sync(0xffffffff, sq, mask);
    }

    if (threadIdx.x == 0) {
        float std_val = sqrtf(sq / (T > 1 ? T - 1 : 1)) + 1e-05f;
        s_inv_std = 1.0f / std_val;
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // --- Pass 3: normalize + write transposed ---
    for (int f = threadIdx.x; f < T; f += blockDim.x)
        mel_out[ch * T + f] = (mel_in[f * 128 + ch] - mean) * inv_std;
}

void mel_normalize(const float* mel_in, float* mel_out,
                   int n_frames, int n_valid, cudaStream_t stream) {
    int threads = n_valid < 1024 ? ((n_valid + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    mel_normalize_kernel<<<128, threads, 0, stream>>>(mel_in, mel_out, n_frames, n_valid);
}

