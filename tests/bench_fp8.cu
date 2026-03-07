// bench_fp8.cu — FP8 vs FP16 microbenchmark for all key encoder GEMM shapes
//
// Compares:
//   - FP16 cublasLt (baseline)
//   - FP8 cublasLt  (per-tensor scaling via A/B scale pointers)
//   - FP8 CUTLASS   (TN layout, 128x128x128 tile, SM120)
//   - FP8 CUTLASS, init cost isolated
//
// Build: make bench_fp8
// Usage: ./bench_fp8

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass_gemm.h"
#include "kernels_fp8.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
struct Timer {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    Timer(cudaStream_t s) : stream(s) {
        cudaEventCreate(&start); cudaEventCreate(&stop);
    }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, stream); }
    float end_us() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms * 1000.0f;
    }
};

// ---------------------------------------------------------------------------
// FP16 cublasLt GEMM (NN row-major: Y[M,N] = X[M,K] @ W[K,N])
// ---------------------------------------------------------------------------
static cublasLtHandle_t g_lt;
static void* g_ws;
static size_t g_ws_size = 32 * 1024 * 1024;

static float bench_fp16_cublas(cudaStream_t stream, Timer& t,
                                 int M, int N, int K,
                                 const half* X, const half* W, half* Y,
                                 int warmup, int iters) {
    // col-major: Y^T[N,M] = W^T[N,K] @ X^T[K,M]  (NN)
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    auto opA = CUBLAS_OP_N, opB = CUBLAS_OP_N;
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    cublasLtMatrixLayout_t lA, lB, lC;
    // col-major: A is W^T[N,K], B is X^T[K,M], C is Y^T[N,M]
    cublasLtMatrixLayoutCreate(&lA, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&lB, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &g_ws_size, sizeof(g_ws_size));
    cublasLtMatmulHeuristicResult_t result;
    int returned;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(g_lt, desc, lA, lB, lC, lC,
                                                  pref, 1, &result, &returned));

    auto run = [&]() {
        cublasLtMatmul(g_lt, desc, &alpha, W, lA, X, lB, &beta, Y, lC, Y, lC,
                       &result.algo, g_ws, g_ws_size, stream);
    };
    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    float us = t.end_us() / iters;

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(desc);
    return us;
}

// ---------------------------------------------------------------------------
// FP8 cublasLt GEMM: quantize X→FP8, then FP8 matmul
// ---------------------------------------------------------------------------
static float bench_fp8_cublas(cudaStream_t stream, Timer& t,
                                int M, int N, int K,
                                const half* X, const uint8_t* W_fp8,
                                const float* w_scale, float* a_scale,
                                uint8_t* act_buf, int* amax_buf,
                                half* Y,
                                int warmup, int iters) {
    // First calibrate to get a_scale
    quantize_absmax_fp16_to_fp8(X, act_buf, a_scale, M * K, amax_buf, stream);
    cudaStreamSynchronize(stream);

    // Build cublasLt FP8 plan
    // col-major layout: A=W^T[N,K], B=act^T[K,M], C=Y^T[N,M]
    auto opA = CUBLAS_OP_N, opB = CUBLAS_OP_N;
    cublasLtMatmulDesc_t desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    const float* w_scale_ptr = w_scale;
    const float* a_scale_ptr = a_scale;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                 &w_scale_ptr, sizeof(w_scale_ptr)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                 &a_scale_ptr, sizeof(a_scale_ptr)));
    cublasLtMatrixLayout_t lA, lB, lC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, N, K, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, N, M, N));
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &g_ws_size, sizeof(g_ws_size)));
    cublasLtMatmulHeuristicResult_t results[4];
    int n_results = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(g_lt, desc, lA, lB, lC, lC,
                                                         pref, 4, results, &n_results);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != CUBLAS_STATUS_SUCCESS || n_results == 0) {
        fprintf(stderr, "FP8 cublasLt: no algo for M=%d N=%d K=%d\n", M, N, K);
        cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
        cublasLtMatmulDescDestroy(desc);
        return -1.0f;
    }
    auto algo = results[0].algo;

    float alpha = 1.0f, beta = 0.0f;
    auto run = [&]() {
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
        cublasLtMatmul(g_lt, desc, &alpha, W_fp8, lA, act_buf, lB, &beta, Y, lC, Y, lC,
                       &algo, g_ws, g_ws_size, stream);
    };
    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    float us = t.end_us() / iters;

    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(desc);
    return us;
}

// ---------------------------------------------------------------------------
// FP8 CUTLASS: quantize X→FP8, then CUTLASS GEMM
// W_fp8 is [N,K] row-major (pre-transposed from [K,N])
// ---------------------------------------------------------------------------
static float bench_fp8_cutlass(cudaStream_t stream, Timer& t,
                                 int M, int N, int K,
                                 const half* X, const uint8_t* W_fp8,
                                 float alpha,
                                 uint8_t* act_buf, float* a_scale,
                                 void* cutlass_ws, size_t cutlass_ws_size,
                                 int warmup, int iters) {
    auto run = [&]() {
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
        cutlass_fp8_gemm(M, N, K, act_buf, W_fp8, (half*)X /* reuse buf */, alpha,
                         stream, cutlass_ws, cutlass_ws_size);
    };

    // Validate
    {
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
        int r = cutlass_fp8_gemm(M, N, K, act_buf, W_fp8, (half*)X, alpha,
                                  stream, cutlass_ws, cutlass_ws_size);
        if (r != 0) {
            fprintf(stderr, "CUTLASS FP8: can_implement/initialize failed for M=%d N=%d K=%d\n",
                    M, N, K);
            return -1.0f;
        }
    }

    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    return t.end_us() / iters;
}

// ---------------------------------------------------------------------------
// Quantize only (to measure its overhead separately)
// ---------------------------------------------------------------------------
static float bench_quantize_only(cudaStream_t stream, Timer& t,
                                   int M, int K,
                                   const half* X, uint8_t* act_buf, float* a_scale,
                                   int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++)
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
    return t.end_us() / iters;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasLtCreate(&g_lt));
    CUDA_CHECK(cudaMalloc(&g_ws, g_ws_size));

    // Allocate buffers (enough for largest shape: 870×4096)
    size_t max_elems = 870 * 4096;
    half*    buf_fp16   = nullptr;
    uint8_t* buf_fp8    = nullptr;
    uint8_t* act_buf    = nullptr;
    half*    out_buf    = nullptr;
    float*   w_scale    = nullptr;
    float*   a_scale    = nullptr;
    int*     amax_buf   = nullptr;
    void*    cutlass_ws = nullptr;

    CUDA_CHECK(cudaMalloc(&buf_fp16, max_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf_fp8,  max_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&act_buf,  max_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&out_buf,  870 * 4096 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w_scale,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a_scale,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_buf, sizeof(int)));

    // Fill FP16 buffer with ~1.0 values
    CUDA_CHECK(cudaMemset(buf_fp16, 0x3C, max_elems * sizeof(half)));
    // Fill FP8 weight with reasonable values
    CUDA_CHECK(cudaMemset(buf_fp8,  0x3C, max_elems * sizeof(uint8_t)));

    // Compute weight scale from absmax
    quantize_absmax_fp16_to_fp8(buf_fp16, buf_fp8, w_scale, 1024 * 4096, amax_buf, stream);
    // Compute act scale
    quantize_absmax_fp16_to_fp8(buf_fp16, act_buf, a_scale, 870 * 1024, amax_buf, stream);
    cudaStreamSynchronize(stream);
    float host_alpha = 1.0f;  // approximate; real alpha = act_scale * wt_scale

    // Get CUTLASS workspace size for max shape
    size_t cutlass_ws_size = cutlass_fp8_workspace_size(870, 4096, 4096);
    if (cutlass_ws_size > 0)
        CUDA_CHECK(cudaMalloc(&cutlass_ws, cutlass_ws_size));

    Timer t(stream);
    int warmup = 20, iters = 200;

    // Shapes: (label, M=T, N, K)
    struct Shape { const char* label; int M, N, K; };
    Shape shapes[] = {
        // FF linear1: Y[T,4096] = X[T,1024] @ W[1024,4096]
        {"FF_lin1",  64,  4096, 1024},
        {"FF_lin1", 128,  4096, 1024},
        {"FF_lin1", 256,  4096, 1024},
        {"FF_lin1", 512,  4096, 1024},
        {"FF_lin1", 870,  4096, 1024},
        // FF linear2: Y[T,1024] = X[T,4096] @ W[4096,1024]
        {"FF_lin2",  64,  1024, 4096},
        {"FF_lin2", 128,  1024, 4096},
        {"FF_lin2", 256,  1024, 4096},
        {"FF_lin2", 512,  1024, 4096},
        {"FF_lin2", 870,  1024, 4096},
        // QKV: Y[T,3072] = X[T,1024] @ W[1024,3072]
        {"QKV",      64,  3072, 1024},
        {"QKV",     128,  3072, 1024},
        {"QKV",     256,  3072, 1024},
        {"QKV",     512,  3072, 1024},
    };

    printf("\n%-14s  %4s  %4s  %4s  %8s  %8s  %8s  %8s  %7s  %7s\n",
           "Shape", "M", "N", "K",
           "FP16(us)", "FP8Lt(us)", "CUTLASS(us)", "Quant(us)",
           "Lt/FP16", "CUT/FP16");
    printf("%-14s  %4s  %4s  %4s  %8s  %8s  %8s  %8s  %7s  %7s\n",
           "--------------", "----", "----", "----",
           "--------", "---------", "-----------", "--------",
           "-------", "-------");

    for (auto& s : shapes) {
        int M = s.M, N = s.N, K = s.K;

        // For W_fp8 we need it in [N,K] layout (pre-transposed for CUTLASS NN)
        // buf_fp8 is already [N,K] (we just use it as-is since it's filled uniformly)
        const half* X = buf_fp16;          // [M,K]
        const uint8_t* W_fp8 = buf_fp8;   // [N,K] for CUTLASS, [K,N] for cublasLt (col-major)

        float fp16_us = bench_fp16_cublas(stream, t, M, N, K, X, buf_fp16, out_buf, warmup, iters);
        float fp8lt_us = bench_fp8_cublas(stream, t, M, N, K, X, buf_fp8, w_scale, a_scale,
                                           act_buf, amax_buf, out_buf, warmup, iters);
        float cutlass_us = bench_fp8_cutlass(stream, t, M, N, K, X, buf_fp8, host_alpha,
                                              act_buf, a_scale, cutlass_ws, cutlass_ws_size,
                                              warmup, iters);
        float quant_us = bench_quantize_only(stream, t, M, K, X, act_buf, a_scale, warmup, iters);

        auto pct = [](float v, float base) -> const char* {
            static char buf[8][16]; static int idx = 0; idx = (idx+1)%8;
            if (v < 0) { snprintf(buf[idx], 16, "   n/a "); return buf[idx]; }
            snprintf(buf[idx], 16, v < base * 0.99f ? " %5.2fx*" : " %5.2fx ",
                     base / v);
            return buf[idx];
        };

        printf("%-14s  %4d  %4d  %4d  %8.1f  %9.1f  %11.1f  %8.1f  %s  %s\n",
               s.label, M, N, K,
               fp16_us, fp8lt_us, cutlass_us, quant_us,
               pct(fp8lt_us, fp16_us), pct(cutlass_us, fp16_us));
    }

    cudaFree(buf_fp16); cudaFree(buf_fp8); cudaFree(act_buf);
    cudaFree(out_buf); cudaFree(w_scale); cudaFree(a_scale);
    cudaFree(amax_buf);
    if (cutlass_ws) cudaFree(cutlass_ws);
    cudaFree(g_ws);
    cublasLtDestroy(g_lt);
    cudaStreamDestroy(stream);
    return 0;
}
