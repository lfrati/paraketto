// bench_ff2.cu — Comprehensive CUTLASS vs cuBLAS survey for FF linear2
//
// Shape: col_m=1024, col_k=4096, col_n=T (varying audio length)
// 48 calls per utterance (24 encoder blocks × 2 FF modules)
//
// Tests: regular tiles × swizzle variants, split-K × slice counts
// Build: make bench_ff2

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/gemm/device/gemm_universal.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s) { \
    fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// =========================================================================
// Type aliases
// =========================================================================

using E  = cutlass::half_t;
using CM = cutlass::layout::ColumnMajor;
using RM = cutlass::layout::RowMajor;
using TensorOp = cutlass::arch::OpClassTensorOp;
using Sm80     = cutlass::arch::Sm80;
using MMA      = cutlass::gemm::GemmShape<16, 8, 16>;

template<int A> using Epi = cutlass::epilogue::thread::LinearCombination<E, A, E, E>;

template<int LogTile>
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<LogTile>;

// --- Regular GEMM (NN col-major) ---
// GemmShape<TbM, TbN, TbK>, WarpShape<WM, WN, WK>, stages, AlignA, AlignB, swizzle

#define DEF_NN(Name, TbM, TbN, TbK, WM, WN, WK, S, Sw, Align) \
    using Name = cutlass::gemm::device::Gemm< \
        E, CM, E, CM, E, CM, E, TensorOp, Sm80, \
        cutlass::gemm::GemmShape<TbM, TbN, TbK>, \
        cutlass::gemm::GemmShape<WM, WN, WK>, MMA, \
        Epi<Align>, Swizzle<Sw>, S, Align, Align>;

// 64x64 family
DEF_NN(NN_64x64_k64_s6,    64, 64, 64, 32,32,64, 6,  1, 8)
DEF_NN(NN_64x64_k64_s6_w4, 64, 64, 64, 32,32,64, 6,  4, 8)
DEF_NN(NN_64x64_k64_s6_w8, 64, 64, 64, 32,32,64, 6,  8, 8)
DEF_NN(NN_64x64_k32_s10,   64, 64, 32, 32,32,32, 10, 1, 8)
DEF_NN(NN_64x64_k32_s6,    64, 64, 32, 32,32,32, 6,  1, 8)

// 128x64 family
DEF_NN(NN_128x64_k32_s5,    128, 64, 32, 64,32,32, 5, 1, 8)
DEF_NN(NN_128x64_k32_s5_w4, 128, 64, 32, 64,32,32, 5, 4, 8)
DEF_NN(NN_128x64_k32_s5_w8, 128, 64, 32, 64,32,32, 5, 8, 8)
DEF_NN(NN_128x64_k64_s4,    128, 64, 64, 64,32,64, 4, 1, 8)
DEF_NN(NN_128x64_k64_s4_w4, 128, 64, 64, 64,32,64, 4, 4, 8)

// 64x128 family
DEF_NN(NN_64x128_k32_s5,    64, 128, 32, 32,64,32, 5, 1, 8)
DEF_NN(NN_64x128_k32_s5_w4, 64, 128, 32, 32,64,32, 5, 4, 8)
DEF_NN(NN_64x128_k64_s4,    64, 128, 64, 32,64,64, 4, 1, 8)

// 128x128 family
DEF_NN(NN_128x128_k32_s5,    128, 128, 32, 64,64,32, 5, 1, 8)
DEF_NN(NN_128x128_k32_s5_w4, 128, 128, 32, 64,64,32, 5, 4, 8)
DEF_NN(NN_128x128_k32_s5_w8, 128, 128, 32, 64,64,32, 5, 8, 8)
DEF_NN(NN_128x128_k32_s3,    128, 128, 32, 64,64,32, 3, 1, 8)

// 256x64 family
DEF_NN(NN_256x64_k32_s3,    256, 64, 32, 64,64,32, 3, 1, 8)
DEF_NN(NN_256x64_k32_s3_w4, 256, 64, 32, 64,64,32, 3, 4, 8)

// 64x256 family
DEF_NN(NN_64x256_k32_s3,    64, 256, 32, 64,64,32, 3, 1, 8)
DEF_NN(NN_64x256_k32_s3_w4, 64, 256, 32, 64,64,32, 3, 4, 8)

// 256x128 / 128x256
DEF_NN(NN_256x128_k32_s3,   256, 128, 32, 64,64,32, 3, 1, 8)
DEF_NN(NN_128x256_k32_s3,   128, 256, 32, 64,64,32, 3, 1, 8)

// --- Split-K (NN col-major) ---
#define DEF_SK(Name, TbM, TbN, TbK, WM, WN, WK, Align) \
    using Name = cutlass::gemm::device::GemmSplitKParallel< \
        E, CM, E, CM, E, CM, E, TensorOp, Sm80, \
        cutlass::gemm::GemmShape<TbM, TbN, TbK>, \
        cutlass::gemm::GemmShape<WM, WN, WK>, MMA, Epi<Align>>;

DEF_SK(SK_64x64_k64,   64,  64, 64, 32,32,64, 8)
DEF_SK(SK_128x64_k32, 128,  64, 32, 64,32,32, 8)
DEF_SK(SK_64x64_k32,   64,  64, 32, 32,32,32, 8)
DEF_SK(SK_128x128_k32, 128, 128, 32, 64,64,32, 8)  // cuBLAS uses this for T=512
DEF_SK(SK_64x128_k32,   64, 128, 32, 32,64,32, 8)
DEF_SK(SK_256x64_k32,  256,  64, 32, 64,64,32, 8)  // cuBLAS uses this for T=260-279 (sk=3)

// --- GemmUniversal with identity swizzle + kGemmSplitKParallel ---
// This should give the correct grid (ceil(M/TbM)*ceil(N/TbN), 1, sk)
// unlike GemmSplitKParallel which uses GemmSplitKHorizontalThreadblockSwizzle
#define DEF_UNIV(Name, TbM, TbN, TbK, WM, WN, WK, S, Align) \
    using Name = cutlass::gemm::device::GemmUniversal< \
        E, CM, E, CM, E, CM, E, TensorOp, Sm80, \
        cutlass::gemm::GemmShape<TbM, TbN, TbK>, \
        cutlass::gemm::GemmShape<WM, WN, WK>, MMA, \
        Epi<Align>, Swizzle<1>, S, Align, Align>;

DEF_UNIV(UNIV_256x64_k32_s3,  256, 64, 32, 64,64,32, 3, 8)
DEF_UNIV(UNIV_256x64_k32_s4,  256, 64, 32, 64,64,32, 4, 8)
DEF_UNIV(UNIV_128x64_k32_s5,  128, 64, 32, 64,32,32, 5, 8)

// =========================================================================
// Timer / cuBLAS helpers
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

static cublasLtHandle_t s_lt;
static void* s_ws;
static size_t s_ws_size = 32 * 1024 * 1024;

static float bench_cublas(cudaStream_t stream, Timer& t,
                           int M, int N, int K,
                           const half* A, int ldA,
                           const half* B, int ldB,
                           half* C, int ldC,
                           int warmup, int iters) {
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
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(s_lt, desc, lA, lB, lC, lC, pref, 1, &result, &returned));

    auto run = [&]() {
        cublasLtMatmul(s_lt, desc, &alpha, A, lA, B, lB, &beta, C, lC, C, lC,
                       &result.algo, s_ws, s_ws_size, stream);
    };
    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    float us = t.end() / iters * 1000.0f;

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(desc);
    return us;
}

// =========================================================================
// Generic benchmarkers
// =========================================================================

template<typename GemmOp>
static float bench_gemm(cudaStream_t stream, Timer& t,
                         int M, int N, int K,
                         const half* A, int ldA,
                         const half* B, int ldB,
                         half* C, int ldC,
                         int warmup, int iters) {
    auto run = [&]() {
        typename GemmOp::Arguments args(
            {M, N, K},
            {(const E*)A, ldA}, {(const E*)B, ldB},
            {(E*)C, ldC}, {(E*)C, ldC},
            {E(1.0f), E(0.0f)});
        GemmOp op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return;
        op.initialize(args, nullptr, stream);
        op(stream);
    };

    // Validate first
    {
        typename GemmOp::Arguments args(
            {M, N, K},
            {(const E*)A, ldA}, {(const E*)B, ldB},
            {(E*)C, ldC}, {(E*)C, ldC},
            {E(1.0f), E(0.0f)});
        GemmOp op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return -1.0f;
    }

    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    return t.end() / iters * 1000.0f;
}

template<typename SplitKOp>
static float bench_splitk(cudaStream_t stream, Timer& t,
                            int M, int N, int K,
                            const half* A, int ldA,
                            const half* B, int ldB,
                            half* C, int ldC,
                            void* ws, int sk,
                            int warmup, int iters) {
    auto run = [&]() {
        typename SplitKOp::Arguments args(
            {M, N, K},
            {(const E*)A, ldA}, {(const E*)B, ldB},
            {(E*)C, ldC}, {(E*)C, ldC},
            {E(1.0f), E(0.0f)}, sk);
        SplitKOp op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return;
        op.initialize(args, ws);
        op.run(stream);
    };

    {
        typename SplitKOp::Arguments args(
            {M, N, K},
            {(const E*)A, ldA}, {(const E*)B, ldB},
            {(E*)C, ldC}, {(E*)C, ldC},
            {E(1.0f), E(0.0f)}, sk);
        SplitKOp op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return -1.0f;
    }

    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    return t.end() / iters * 1000.0f;
}

// =========================================================================
// GemmUniversal split-K benchmarker (uses GemmIdentityThreadblockSwizzle)
// =========================================================================

template<typename UnivOp>
static float bench_universal_splitk(cudaStream_t stream, Timer& t,
                                     int M, int N, int K,
                                     const half* A, int ldA,
                                     const half* B, int ldB,
                                     half* C, int ldC,
                                     void* ws, int sk,
                                     int warmup, int iters) {
    // GemmUniversal with kGemmSplitKParallel: batch_count = sk slices
    auto make_args = [&]() {
        return typename UnivOp::Arguments(
            cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
            {M, N, K},
            sk,
            {E(1.0f), E(0.0f)},
            (const void*)A, (const void*)B, (const void*)C, (void*)C,
            (int64_t)0, (int64_t)0, (int64_t)0, (int64_t)0,
            ldA, ldB, ldC, ldC);
    };

    {
        auto args = make_args();
        UnivOp op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return -1.0f;
        size_t ws_sz = UnivOp::get_workspace_size(args);
        if (ws_sz > 32u * 1024 * 1024) return -2.0f;
    }

    auto run = [&]() {
        auto args = make_args();
        UnivOp op;
        size_t ws_sz = UnivOp::get_workspace_size(args);
        op.initialize(args, ws_sz > 0 ? ws : nullptr, stream);
        op.run(stream);
    };

    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    return t.end() / iters * 1000.0f;
}

// =========================================================================
// Per-T result tracking
// =========================================================================

struct Result {
    const char* name;
    float us;
};

static Result s_results[64];
static int s_nresults;

static void record(const char* name, float us, float cublas_us) {
    if (us < 0) return;  // not implemented
    s_results[s_nresults++] = {name, us};
    float ratio = us / cublas_us;
    printf("  %-30s  %6.1f us  (%+5.1f%%)%s\n", name, us,
           (ratio - 1.0f) * 100.0f,
           ratio < 0.98f ? "  <<<<" : "");
}

// =========================================================================
// Main
// =========================================================================

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasLtCreate(&s_lt));
    CUDA_CHECK(cudaMalloc(&s_ws, s_ws_size));

    void* sk_ws;
    CUDA_CHECK(cudaMalloc(&sk_ws, 32 * 1024 * 1024));

    half* buf;
    CUDA_CHECK(cudaMalloc(&buf, 256 * 1024 * 1024));
    CUDA_CHECK(cudaMemset(buf, 0x3C, 256 * 1024 * 1024));

    Timer t(stream);
    int warmup = 100, iters = 500;

    // Full sweep: UNIV configs vs cuBLAS for T=129-512
    int T_values[] = {129, 144, 160, 176, 192, 208, 224, 240, 256,
                      260, 272, 288, 320, 352, 384, 416, 448, 480, 512};
    int nT = sizeof(T_values) / sizeof(T_values[0]);

    printf("\nFull T sweep with GemmUniversal  M=1024, K=4096   (* = beats cuBLAS >1%%)\n");
    printf("  %-5s  %8s  %8s  %8s  %8s  %8s\n",
           "T", "cuBLAS", "64x64_s6", "UNIV256sk4", "UNIV128sk4", "64x128s5");

    for (int ti = 0; ti < nT; ti++) {
        int T = T_values[ti];
        int col_m = 1024, col_n = T, col_k = 4096;
        int ldA = col_m, ldB = col_k, ldC = col_m;
        half* A = buf, *B = buf + col_m*col_k, *C = buf + col_m*col_k + col_k*col_n;

        float cub    = bench_cublas(stream, t, col_m,col_n,col_k, A,ldA,B,ldB,C,ldC, warmup,iters);
        float r64    = bench_gemm<NN_64x64_k64_s6>(stream, t, col_m,col_n,col_k, A,ldA,B,ldB,C,ldC, warmup,iters);
        float u256s4 = bench_universal_splitk<UNIV_256x64_k32_s4>(stream, t, col_m,col_n,col_k, A,ldA,B,ldB,C,ldC, sk_ws, 4, warmup,iters);
        float u128s4 = bench_universal_splitk<UNIV_128x64_k32_s5>(stream, t, col_m,col_n,col_k, A,ldA,B,ldB,C,ldC, sk_ws, 4, warmup,iters);
        float r64x128 = bench_gemm<NN_64x128_k32_s5>(stream, t, col_m,col_n,col_k, A,ldA,B,ldB,C,ldC, warmup,iters);

        auto fmt = [&](float v) -> const char* {
            static char buf[8][16]; static int idx=0;
            idx=(idx+1)%8;
            if (v < 0) { snprintf(buf[idx], 16, "     n/a  "); return buf[idx]; }
            snprintf(buf[idx], 16, v < cub*0.99f ? "*%5.1fus" : " %6.1fus", v);
            return buf[idx];
        };
        printf("  %-5d  %7.1fus %s %s %s %s\n",
               T, cub, fmt(r64), fmt(u256s4), fmt(u128s4), fmt(r64x128));
    }

    cudaFree(buf); cudaFree(sk_ws); cudaFree(s_ws);
    cublasLtDestroy(s_lt); cudaStreamDestroy(stream);
    return 0;
}
