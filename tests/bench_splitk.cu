// bench_splitk.cu — Find optimal split-K factor for FF linear2 shape
//
// Build: make bench_splitk
// Usage: ./bench_splitk

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s) { fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// =========================================================================
// CUTLASS types
// =========================================================================

template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using SplitKSwizzle = cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle;

// Current config (no split-K)
using GemmNN_64x64_64_s6 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 6, 8, 8>;

// Split-K parallel version — same tile config
using GemmSplitK_64x64_64_s6 = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>>;

// Split-K with 32-wide K tile (more slices possible)
using GemmSplitK_64x64_32_s6 = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>>;

// =========================================================================
// Timing helper
// =========================================================================

struct Timer {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    Timer(cudaStream_t s) : stream(s) {
        cudaEventCreate(&start); cudaEventCreate(&stop);
    }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, stream); }
    float end() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// =========================================================================
// cuBLAS baseline
// =========================================================================

static cublasLtHandle_t s_cublaslt;
static void* s_cublas_ws;
static size_t s_cublas_ws_size = 32 * 1024 * 1024;

void cublas_nn(cudaStream_t stream, int M, int N, int K,
               const half* A, int ldA, const half* B, int ldB,
               half* C, int ldC) {
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    auto opA = CUBLAS_OP_N, opB = CUBLAS_OP_N;

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    cublasLtMatrixLayout_t lA, lB, lC;
    cublasLtMatrixLayoutCreate(&lA, CUDA_R_16F, M, K, ldA);
    cublasLtMatrixLayoutCreate(&lB, CUDA_R_16F, K, N, ldB);
    cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, M, N, ldC);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &s_cublas_ws_size, sizeof(s_cublas_ws_size));

    cublasLtMatmulHeuristicResult_t result;
    int returned;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(s_cublaslt, desc, lA, lB, lC, lC,
                                    pref, 1, &result, &returned));

    CUBLAS_CHECK(cublasLtMatmul(s_cublaslt, desc, &alpha, A, lA, B, lB,
                   &beta, C, lC, C, lC,
                   &result.algo, s_cublas_ws, s_cublas_ws_size, stream));

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA);
    cublasLtMatrixLayoutDestroy(lB);
    cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(desc);
}

// =========================================================================
// Main
// =========================================================================

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUBLAS_CHECK(cublasLtCreate(&s_cublaslt));
    CUDA_CHECK(cudaMalloc(&s_cublas_ws, s_cublas_ws_size));

    half* buf;
    CUDA_CHECK(cudaMalloc(&buf, 128 * 1024 * 1024));
    CUDA_CHECK(cudaMemset(buf, 0x3C, 128 * 1024 * 1024));

    // Split-K workspace
    void* splitk_ws;
    CUDA_CHECK(cudaMalloc(&splitk_ws, 32 * 1024 * 1024));

    Timer t(stream);
    int warmup = 100, iters = 500;

    // Test multiple T values
    int T_values[] = {63, 125, 188, 375, 500, 750};

    for (int T : T_values) {
        // FF linear2: row-major Y[T,1024] = X[T,4096] @ W[4096,1024]
        // col-major: col_m=1024, col_n=T, col_k=4096
        int col_m = 1024, col_n = T, col_k = 4096;
        int ldA = col_m, ldB = col_k, ldC = col_m;

        half* A = buf;
        half* B = buf + col_m * col_k;
        half* C = buf + col_m * col_k + col_k * col_n;

        printf("\n--- FF linear2: col %dx%dx%d (T=%d) ---\n", col_m, col_n, col_k, T);

        // cuBLAS baseline
        for (int i = 0; i < warmup; i++)
            cublas_nn(stream, col_m, col_n, col_k, A, ldA, B, ldB, C, ldC);
        cudaStreamSynchronize(stream);
        t.begin();
        for (int i = 0; i < iters; i++)
            cublas_nn(stream, col_m, col_n, col_k, A, ldA, B, ldB, C, ldC);
        float cublas_ms = t.end() / iters;
        printf("  cuBLAS:              %.3f us\n", cublas_ms * 1000);

        // Current CUTLASS (no split-K)
        {
            using E = cutlass::half_t;
            auto run = [&]() {
                typename GemmNN_64x64_64_s6::Arguments args(
                    {col_m, col_n, col_k},
                    {(const E*)A, ldA}, {(const E*)B, ldB},
                    {(E*)C, ldC}, {(E*)C, ldC}, {E(1.0f), E(0.0f)});
                GemmNN_64x64_64_s6 op;
                op.initialize(args, nullptr, stream);
                op(stream);
            };
            for (int i = 0; i < warmup; i++) run();
            cudaStreamSynchronize(stream);
            t.begin();
            for (int i = 0; i < iters; i++) run();
            float ms = t.end() / iters;
            printf("  CUTLASS (no splitK): %.3f us  (%.2fx cuBLAS)\n", ms * 1000, ms / cublas_ms);
        }

        // Split-K with K=64 tile
        for (int sk : {2, 4, 8, 16}) {
            using E = cutlass::half_t;
            auto run = [&]() {
                typename GemmSplitK_64x64_64_s6::Arguments args(
                    {col_m, col_n, col_k},
                    {(const E*)A, ldA}, {(const E*)B, ldB},
                    {(E*)C, ldC}, {(E*)C, ldC}, {E(1.0f), E(0.0f)}, sk);
                GemmSplitK_64x64_64_s6 op;
                size_t ws_need = GemmSplitK_64x64_64_s6::get_workspace_size(args);
                auto status = op.initialize(args, splitk_ws);
                if (status != cutlass::Status::kSuccess) {
                    printf("  SplitK k64 sk=%d: init failed\n", sk);
                    return;
                }
                op.run(stream);
            };
            for (int i = 0; i < warmup; i++) run();
            cudaStreamSynchronize(stream);
            t.begin();
            for (int i = 0; i < iters; i++) run();
            float ms = t.end() / iters;
            printf("  SplitK k64 sk=%-2d:   %.3f us  (%.2fx cuBLAS)\n", sk, ms * 1000, ms / cublas_ms);
        }

        // Split-K with K=32 tile
        for (int sk : {2, 4, 8, 16}) {
            using E = cutlass::half_t;
            auto run = [&]() {
                typename GemmSplitK_64x64_32_s6::Arguments args(
                    {col_m, col_n, col_k},
                    {(const E*)A, ldA}, {(const E*)B, ldB},
                    {(E*)C, ldC}, {(E*)C, ldC}, {E(1.0f), E(0.0f)}, sk);
                GemmSplitK_64x64_32_s6 op;
                auto status = op.initialize(args, splitk_ws);
                if (status != cutlass::Status::kSuccess) {
                    printf("  SplitK k32 sk=%d: init failed\n", sk);
                    return;
                }
                op.run(stream);
            };
            for (int i = 0; i < warmup; i++) run();
            cudaStreamSynchronize(stream);
            t.begin();
            for (int i = 0; i < iters; i++) run();
            float ms = t.end() / iters;
            printf("  SplitK k32 sk=%-2d:   %.3f us  (%.2fx cuBLAS)\n", sk, ms * 1000, ms / cublas_ms);
        }
    }

    cudaFree(buf);
    cudaFree(splitk_ws);
    cudaFree(s_cublas_ws);
    cublasLtDestroy(s_cublaslt);
    cudaStreamDestroy(stream);
    return 0;
}
