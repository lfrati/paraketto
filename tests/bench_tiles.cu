// bench_tiles.cu — Find optimal CUTLASS tile config for FF linear2 at large T
//
// The shape: col_m=1024, col_n=T, col_k=4096 (NN, ColMajor)
// cuBLAS wins at T>=250. Try all reasonable tile configs.

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

template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Current default
using Tile_64x64_k64_s6 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 6, 8, 8>;

// Try 128x64 tile
using Tile_128x64_k32_s5 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 5, 8, 8>;

// Try 128x128 tile
using Tile_128x128_k32_s5 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 5, 8, 8>;

// Try 256x64 tile
using Tile_256x64_k32_s3 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 3, 8, 8>;

// Try 64x128 tile
using Tile_64x128_k32_s5 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 5, 8, 8>;

// Try 128x64 with k=64
using Tile_128x64_k64_s4 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 4, 8, 8>;

// Try 256x128 tile
using Tile_256x128_k32_s3 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 3, 8, 8>;

// Try 128x256 tile
using Tile_128x256_k32_s3 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 3, 8, 8>;

// =========================================================================

struct Timer {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    Timer(cudaStream_t s) : stream(s) { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, stream); }
    float end() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// cuBLAS baseline
static cublasLtHandle_t s_cublaslt;
static void* s_ws;
static size_t s_ws_size = 32 * 1024 * 1024;

void cublas_nn(cudaStream_t stream, int M, int N, int K,
               const half* A, int ldA, const half* B, int ldB, half* C, int ldC) {
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
                                          &s_ws_size, sizeof(s_ws_size));
    cublasLtMatmulHeuristicResult_t result;
    int returned;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(s_cublaslt, desc, lA, lB, lC, lC, pref, 1, &result, &returned));
    CUBLAS_CHECK(cublasLtMatmul(s_cublaslt, desc, &alpha, A, lA, B, lB, &beta, C, lC, C, lC,
                   &result.algo, s_ws, s_ws_size, stream));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(desc);
}

template<typename GemmOp>
static float bench_config(const char* name, cudaStream_t stream, Timer& t,
                           int M, int N, int K, const half* A, int ldA,
                           const half* B, int ldB, half* C, int ldC,
                           float cublas_us, int warmup, int iters) {
    using E = cutlass::half_t;
    auto run = [&]() {
        typename GemmOp::Arguments args(
            {M, N, K},
            {(const E*)A, ldA}, {(const E*)B, ldB},
            {(E*)C, ldC}, {(E*)C, ldC}, {E(1.0f), E(0.0f)});
        GemmOp op;
        auto s = op.can_implement(args);
        if (s != cutlass::Status::kSuccess) return;
        op.initialize(args, nullptr, stream);
        op(stream);
    };

    // Test if config is valid
    {
        typename GemmOp::Arguments args(
            {M, N, K},
            {(const E*)A, ldA}, {(const E*)B, ldB},
            {(E*)C, ldC}, {(E*)C, ldC}, {E(1.0f), E(0.0f)});
        GemmOp op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) {
            printf("  %-28s  N/A (can't implement)\n", name);
            return 999.0f;
        }
    }

    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    float ms = t.end() / iters;
    float us = ms * 1000;
    float ratio = us / cublas_us;
    printf("  %-28s  %7.3f us  (%.2fx cuBLAS)%s\n", name, us, ratio,
           ratio < 0.98f ? "  <<<" : ratio > 1.02f ? "" : "  ===");
    return ratio;
}

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasLtCreate(&s_cublaslt));
    CUDA_CHECK(cudaMalloc(&s_ws, s_ws_size));

    half* buf;
    CUDA_CHECK(cudaMalloc(&buf, 256 * 1024 * 1024));
    CUDA_CHECK(cudaMemset(buf, 0x3C, 256 * 1024 * 1024));

    Timer t(stream);
    int warmup = 100, iters = 500;

    int T_values[] = {63, 125, 188, 250, 375, 500, 625, 750};

    for (int T : T_values) {
        int col_m = 1024, col_n = T, col_k = 4096;
        int ldA = col_m, ldB = col_k, ldC = col_m;
        half* A = buf;
        half* B = buf + col_m * col_k;
        half* C = buf + col_m * col_k + col_k * col_n;

        // cuBLAS baseline
        for (int i = 0; i < warmup; i++)
            cublas_nn(stream, col_m, col_n, col_k, A, ldA, B, ldB, C, ldC);
        cudaStreamSynchronize(stream);
        t.begin();
        for (int i = 0; i < iters; i++)
            cublas_nn(stream, col_m, col_n, col_k, A, ldA, B, ldB, C, ldC);
        float cublas_us = t.end() / iters * 1000;

        printf("\n--- T=%d (col %dx%dx%d) — cuBLAS: %.3f us ---\n", T, col_m, col_n, col_k, cublas_us);

        bench_config<Tile_64x64_k64_s6>("64x64 k64 s6 (current)", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
        bench_config<Tile_128x64_k32_s5>("128x64 k32 s5", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
        bench_config<Tile_128x64_k64_s4>("128x64 k64 s4", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
        bench_config<Tile_64x128_k32_s5>("64x128 k32 s5", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
        bench_config<Tile_128x128_k32_s5>("128x128 k32 s5", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
        bench_config<Tile_256x64_k32_s3>("256x64 k32 s3", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
        bench_config<Tile_256x128_k32_s3>("256x128 k32 s3", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
        bench_config<Tile_128x256_k32_s3>("128x256 k32 s3", stream, t,
            col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, cublas_us, warmup, iters);
    }

    cudaFree(buf);
    cudaFree(s_ws);
    cublasLtDestroy(s_cublaslt);
    cudaStreamDestroy(stream);
    return 0;
}
