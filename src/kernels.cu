// kernels.cu — Custom CUDA kernels for Parakeet conformer
//
// All kernels use FP16 I/O with FP32 accumulation where needed.
// Block/grid sizes tuned for typical conformer dimensions (D=1024, H=640).

#include "kernels.h"
#include <cfloat>

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
// LayerNorm + residual: y = x_residual + alpha * LN(x)
// ---------------------------------------------------------------------------

__global__ void layer_norm_residual_kernel(const half* __restrict__ x,
                                           const half* __restrict__ x_res,
                                           const half* __restrict__ gamma,
                                           const half* __restrict__ beta,
                                           half* __restrict__ y,
                                           int N, int D, float eps, float alpha) {
    int row = blockIdx.x;
    if (row >= N) return;

    const half* xr = x + row * D;
    const half* rr = x_res + row * D;
    half* yr = y + row * D;

    float sum = 0.0f, sum2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        sum += v;
        sum2 += v * v;
    }

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

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        float ln = (v - mean) * inv_std * g + b;
        float r = __half2float(rr[i]);
        yr[i] = __float2half(r + alpha * ln);
    }
}

void layer_norm_residual_fp16(const half* x, const half* x_residual,
                              const half* gamma, const half* beta,
                              half* y, int N, int D, float eps, float alpha,
                              cudaStream_t stream) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    layer_norm_residual_kernel<<<N, threads, 0, stream>>>(
        x, x_residual, gamma, beta, y, N, D, eps, alpha);
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
// SiLU
// ---------------------------------------------------------------------------

__global__ void silu_kernel(const half* __restrict__ x, half* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        y[i] = __float2half(v / (1.0f + expf(-v)));
    }
}

void silu_fp16(const half* x, half* y, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}

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
// Bias add
// ---------------------------------------------------------------------------

__global__ void bias_add_kernel(const half* __restrict__ x,
                                const half* __restrict__ bias,
                                half* __restrict__ y,
                                int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D;
    if (idx < total) {
        float v = __half2float(x[idx]) + __half2float(bias[idx % D]);
        y[idx] = __float2half(v);
    }
}

void bias_add_fp16(const half* x, const half* bias, half* y,
                   int N, int D, cudaStream_t stream) {
    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_kernel<<<blocks, threads, 0, stream>>>(x, bias, y, N, D);
}

__global__ void bias_add_inplace_kernel(half* __restrict__ x,
                                        const half* __restrict__ bias,
                                        int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D;
    if (idx < total) {
        float v = __half2float(x[idx]) + __half2float(bias[idx % D]);
        x[idx] = __float2half(v);
    }
}

void bias_add_inplace_fp16(half* x, const half* bias, int N, int D,
                           cudaStream_t stream) {
    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_inplace_kernel<<<blocks, threads, 0, stream>>>(x, bias, N, D);
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
// Depthwise conv 1D, kernel=9, stride=1, padding=4
// Layout: x[T, C], w[C, 1, 9], b[C], y[T, C]
// Each thread handles one (t, c) output element.
// ---------------------------------------------------------------------------

__global__ void depthwise_conv1d_k9_kernel(const half* __restrict__ x,
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
        y[idx] = __float2half(sum);
    }
}

void depthwise_conv1d_k9_fp16(const half* x, const half* w, const half* b,
                              half* y, int T, int C, cudaStream_t stream) {
    int total = T * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    depthwise_conv1d_k9_kernel<<<blocks, threads, 0, stream>>>(x, w, b, y, T, C);
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
// Softmax
// ---------------------------------------------------------------------------

__global__ void softmax_kernel(const half* __restrict__ x, half* __restrict__ y,
                               int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* xr = x + row * cols;
    half* yr = y + row * cols;

    // Find max
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __half2float(xr[i]);
        if (v > max_val) max_val = v;
    }
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

    // Sum exp
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        sum += expf(__half2float(xr[i]) - max_val);
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
    __shared__ float s_sum_final;
    if (threadIdx.x == 0) s_sum_final = 1.0f / sum;
    __syncthreads();
    float inv_sum = s_sum_final;

    // Write output
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        yr[i] = __float2half(expf(__half2float(xr[i]) - max_val) * inv_sum);
}

void softmax_fp16(const half* x, half* y, int rows, int cols,
                  cudaStream_t stream) {
    int threads = (cols <= 1024) ? ((cols + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    softmax_kernel<<<rows, threads, 0, stream>>>(x, y, rows, cols);
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
// Embedding gather
// ---------------------------------------------------------------------------

__global__ void embedding_gather_kernel(const half* __restrict__ table,
                                        int idx, half* __restrict__ y, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) {
        y[i] = table[idx * D + i];
    }
}

void embedding_gather_fp16(const half* table, int idx, half* y, int D,
                           cudaStream_t stream) {
    int threads = 256;
    int blocks = (D + threads - 1) / threads;
    embedding_gather_kernel<<<blocks, threads, 0, stream>>>(table, idx, y, D);
}

// ---------------------------------------------------------------------------
// FP16 <-> FP32 casts
// ---------------------------------------------------------------------------

__global__ void cast_fp16_to_fp32_kernel(const half* __restrict__ x,
                                         float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

void cast_fp16_to_fp32(const half* x, float* y, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cast_fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}

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
// Add position bias: q_out[h, t, d] = q[h, t, d] + pos_bias[h, d]
// ---------------------------------------------------------------------------

__global__ void add_pos_bias_kernel(const half* __restrict__ q,
                                    const half* __restrict__ pos_bias,
                                    half* __restrict__ q_out,
                                    int heads, int T, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * T * head_dim;
    if (idx < total) {
        int h = idx / (T * head_dim);
        int d = idx % head_dim;
        float v = __half2float(q[idx]) + __half2float(pos_bias[h * head_dim + d]);
        q_out[idx] = __float2half(v);
    }
}

void add_pos_bias_fp16(const half* q, const half* pos_bias,
                       half* q_out, int heads, int T, int head_dim,
                       cudaStream_t stream) {
    int total = heads * T * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_pos_bias_kernel<<<blocks, threads, 0, stream>>>(
        q, pos_bias, q_out, heads, T, head_dim);
}

// ---------------------------------------------------------------------------
// Dual add position bias: q_u = q + bias_u, q_v = q + bias_v
// Reads Q once, writes both outputs.
// ---------------------------------------------------------------------------

__global__ void add_pos_bias_dual_kernel(const half* __restrict__ q,
                                          const half* __restrict__ bias_u,
                                          const half* __restrict__ bias_v,
                                          half* __restrict__ q_u,
                                          half* __restrict__ q_v,
                                          int heads, int T, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * T * head_dim;
    if (idx < total) {
        int h = idx / (T * head_dim);
        int d = idx % head_dim;
        float qv = __half2float(q[idx]);
        float bu = __half2float(bias_u[h * head_dim + d]);
        float bv = __half2float(bias_v[h * head_dim + d]);
        q_u[idx] = __float2half(qv + bu);
        q_v[idx] = __float2half(qv + bv);
    }
}

void add_pos_bias_dual_fp16(const half* q,
                             const half* bias_u, const half* bias_v,
                             half* q_u, half* q_v,
                             int heads, int T, int head_dim,
                             cudaStream_t stream) {
    int total = heads * T * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_pos_bias_dual_kernel<<<blocks, threads, 0, stream>>>(
        q, bias_u, bias_v, q_u, q_v, heads, T, head_dim);
}

// ---------------------------------------------------------------------------
// Relative position skew: [heads, T, 2T-1] -> [heads, T, T]
// The Transformer-XL trick: pad, reshape, slice.
// For each head h and row t, we extract columns [T-1-t .. 2T-2-t] from
// the input [t, 0..2T-2] to get the aligned position scores.
// ---------------------------------------------------------------------------

__global__ void rel_pos_skew_kernel(const half* __restrict__ in,
                                    half* __restrict__ out,
                                    int heads, int T, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * T * T;
    if (idx < total) {
        int h = idx / (T * T);
        int rem = idx % (T * T);
        int t = rem / T;
        int j = rem % T;
        // Input column for position (t, j): j + (T - 1 - t)
        int src_col = j + T - 1 - t;
        int W = 2 * T - 1;
        float val = __half2float(in[h * T * W + t * W + src_col]);
        out[idx] = __float2half(val * scale);
    }
}

void rel_pos_skew_fp16(const half* pos_scores, half* out,
                       int heads, int T, cudaStream_t stream) {
    int total = heads * T * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rel_pos_skew_kernel<<<blocks, threads, 0, stream>>>(pos_scores, out, heads, T, 1.0f);
}

void rel_pos_skew_scale_fp16(const half* pos_scores, half* out,
                              int heads, int T, float scale, cudaStream_t stream) {
    int total = heads * T * T;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rel_pos_skew_kernel<<<blocks, threads, 0, stream>>>(pos_scores, out, heads, T, scale);
}

// ---------------------------------------------------------------------------
// Scale + add scores: out = (content + position) * scale
// ---------------------------------------------------------------------------

__global__ void scale_add_scores_kernel(const half* __restrict__ content,
                                        const half* __restrict__ position,
                                        half* __restrict__ out,
                                        int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = (__half2float(content[i]) + __half2float(position[i])) * scale;
        out[i] = __float2half(v);
    }
}

void scale_add_scores_fp16(const half* content, const half* position,
                           half* out, int heads, int T, float scale,
                           cudaStream_t stream) {
    int n = heads * T * T;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_add_scores_kernel<<<blocks, threads, 0, stream>>>(
        content, position, out, n, scale);
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
// ReLU in-place
// ---------------------------------------------------------------------------

__global__ void relu_inplace_kernel(half* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        if (v < 0.0f) x[i] = __float2half(0.0f);
    }
}

void relu_inplace_fp16(half* x, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_inplace_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

// ---------------------------------------------------------------------------
// Reshape [C, H, W] -> [W, C*H]
// ---------------------------------------------------------------------------

__global__ void reshape_chw_to_wch_kernel(const half* __restrict__ in,
                                           half* __restrict__ out,
                                           int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;
    int w = idx / (C * H);
    int ch = idx % (C * H);
    int c = ch / H;
    int h = ch % H;
    out[idx] = in[c * H * W + h * W + w];
}

void reshape_chw_to_wch_fp16(const half* in, half* out,
                              int C, int H, int W, cudaStream_t stream) {
    int total = C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_chw_to_wch_kernel<<<blocks, threads, 0, stream>>>(in, out, C, H, W);
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
// Split + transpose: [T, 3*heads*head_dim] -> 3x [heads, T, head_dim]
// ---------------------------------------------------------------------------

__global__ void split_transpose_3way_kernel(const half* __restrict__ in,
                                             half* __restrict__ out0,
                                             half* __restrict__ out1,
                                             half* __restrict__ out2,
                                             int T, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int D = heads * head_dim;
    int total = T * D;  // each output has T*D elements
    if (idx < total) {
        int t = idx / D;
        int rem = idx % D;
        int h = rem / head_dim;
        int d = rem % head_dim;
        // Input: [T, 3*D], at row t, Q starts at col 0, K at D, V at 2*D
        int in_row = t * 3 * D;
        // Output: [heads, T, head_dim], element (h, t, d) at h*T*head_dim + t*head_dim + d
        int out_idx = h * T * head_dim + t * head_dim + d;
        out0[out_idx] = in[in_row + rem];           // Q
        out1[out_idx] = in[in_row + D + rem];       // K
        out2[out_idx] = in[in_row + 2 * D + rem];   // V
    }
}

void split_transpose_3way_fp16(const half* in,
                                half* out0, half* out1, half* out2,
                                int T, int heads, int head_dim,
                                cudaStream_t stream) {
    int total = T * heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    split_transpose_3way_kernel<<<blocks, threads, 0, stream>>>(
        in, out0, out1, out2, T, heads, head_dim);
}

