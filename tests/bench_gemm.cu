// bench_gemm.cu — Head-to-head CUTLASS vs cuBLAS for every encoder GEMM shape
//
// Build: make bench_gemm
// Usage: ./bench_gemm

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s) { fprintf(stderr, "cuBLAS %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// =========================================================================
// CUTLASS kernel types (same as cutlass_gemm.cu)
// =========================================================================

template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

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

using GemmNN_64x64_32_s10 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 10, 8, 8>;

using GemmTN_64x64_64_s6 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 6, 8, 8>;

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
// cuBLAS GEMM wrapper
// =========================================================================

static cublasHandle_t s_cublas;
static cublasLtHandle_t s_cublaslt;
static void* s_workspace;
static size_t s_workspace_size = 32 * 1024 * 1024;

void cublas_gemm(cudaStream_t stream, cublasOperation_t opA, cublasOperation_t opB,
                 int M, int N, int K,
                 const half* A, int ldA, const half* B, int ldB,
                 half* C, int ldC) {
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    int rowA = (opA == CUBLAS_OP_N) ? M : K;
    int colA = (opA == CUBLAS_OP_N) ? K : M;
    int rowB = (opB == CUBLAS_OP_N) ? K : N;
    int colB = (opB == CUBLAS_OP_N) ? N : K;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, rowA, colA, ldA);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16F, rowB, colB, ldB);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, M, N, ldC);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &s_workspace_size, sizeof(s_workspace_size));

    cublasLtMatmulHeuristicResult_t result;
    int returned;
    cublasLtMatmulAlgoGetHeuristic(s_cublaslt, desc, layoutA, layoutB, layoutC, layoutC,
                                    pref, 1, &result, &returned);

    cublasLtMatmul(s_cublaslt, desc, &alpha, A, layoutA, B, layoutB,
                   &beta, C, layoutC, C, layoutC,
                   &result.algo, s_workspace, s_workspace_size, stream);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(desc);
}

// =========================================================================
// CUTLASS GEMM wrapper
// =========================================================================

template<typename GemmOp>
static void cutlass_gemm(int M, int N, int K,
                          const half* A, int ldA, const half* B, int ldB,
                          half* C, int ldC, cudaStream_t stream) {
    using E = cutlass::half_t;
    typename GemmOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const E*>(A), ldA},
        {reinterpret_cast<const E*>(B), ldB},
        {reinterpret_cast<E*>(C), ldC},
        {reinterpret_cast<E*>(C), ldC},
        {E(1.0f), E(0.0f)}
    );
    GemmOp op;
    op.initialize(args, nullptr, stream);
    op(stream);
}

// =========================================================================
// Benchmark one shape
// =========================================================================

struct Shape {
    const char* name;
    int m, n, k;        // row-major: Y[m,n] = X[m,k] @ W[k,n] (NN) or W[n,k]^T (NT)
    bool transB;        // false=NN, true=NT
};

void bench_shape(cudaStream_t stream, const Shape& s, half* buf, int warmup, int iters) {
    // col-major params
    int col_m = s.n, col_n = s.m, col_k = s.k;
    int ldA, ldB, ldC;
    cublasOperation_t opA, opB;

    if (!s.transB) {
        // NN: col A=W[n,k]→ColMaj, B=X[k,m]→ColMaj
        ldA = s.n; ldB = s.k; ldC = s.n;
        opA = CUBLAS_OP_N; opB = CUBLAS_OP_N;
    } else {
        // NT: col A=W[k,n]→RowMaj, B=X[k,m]→ColMaj
        ldA = s.k; ldB = s.k; ldC = s.n;
        opA = CUBLAS_OP_T; opB = CUBLAS_OP_N;
    }

    half* A = buf;
    half* B = buf + col_m * col_k;
    half* C = buf + col_m * col_k + col_k * col_n;

    Timer t(stream);

    // --- cuBLAS ---
    for (int i = 0; i < warmup; i++)
        cublas_gemm(stream, opA, opB, col_m, col_n, col_k, A, ldA, B, ldB, C, ldC);
    cudaStreamSynchronize(stream);

    t.begin();
    for (int i = 0; i < iters; i++)
        cublas_gemm(stream, opA, opB, col_m, col_n, col_k, A, ldA, B, ldB, C, ldC);
    float cublas_ms = t.end() / iters;

    // --- CUTLASS ---
    auto run_cutlass = [&]() {
        if (!s.transB) {
            if (col_m == 3072 || (col_m <= 1024 && col_n >= 128 && col_k == 256))
                cutlass_gemm<GemmNN_64x64_32_s10>(col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, stream);
            else
                cutlass_gemm<GemmNN_64x64_64_s6>(col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, stream);
        } else {
            cutlass_gemm<GemmTN_64x64_64_s6>(col_m, col_n, col_k, A, ldA, B, ldB, C, ldC, stream);
        }
    };

    for (int i = 0; i < warmup; i++) run_cutlass();
    cudaStreamSynchronize(stream);

    t.begin();
    for (int i = 0; i < iters; i++) run_cutlass();
    float cutlass_ms = t.end() / iters;

    float ratio = cutlass_ms / cublas_ms;
    const char* verdict = ratio < 0.98f ? "CUTLASS wins" : ratio > 1.02f ? "cuBLAS wins" : "tie";
    printf("%-25s %4dx%-4dx%-4d %s  cublas=%.3fus  cutlass=%.3fus  ratio=%.2f  %s\n",
           s.name, s.m, s.n, s.k, s.transB ? "NT" : "NN",
           cublas_ms * 1000, cutlass_ms * 1000, ratio, verdict);
}

// =========================================================================
// Main
// =========================================================================

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUBLAS_CHECK(cublasCreate(&s_cublas));
    CUBLAS_CHECK(cublasSetStream(s_cublas, stream));
    CUBLAS_CHECK(cublasLtCreate(&s_cublaslt));
    CUDA_CHECK(cudaMalloc(&s_workspace, s_workspace_size));

    // Allocate enough for the largest shape
    half* buf;
    CUDA_CHECK(cudaMalloc(&buf, 256 * 1024 * 1024));  // 256MB
    // Fill with random-ish data
    CUDA_CHECK(cudaMemset(buf, 0x3C, 256 * 1024 * 1024));  // ~1.0 in FP16

    int T = 63;   // typical for ~10s audio
    int pos_len = 2 * T - 1;

    // Per-block shapes (×24)
    Shape shapes[] = {
        {"FF1 linear1",     T, 4096, 1024, false},
        {"FF1 linear2",     T, 1024, 4096, false},   // known gap
        {"Fused QKV",       T, 3072, 1024, false},
        {"Pos proj",        pos_len, 1024, 1024, false},
        {"Out proj",        T, 1024, 1024, false},
        {"Conv pw1",        T, 2048, 1024, true},
        {"Conv pw2",        T, 1024, 1024, true},
        {"FF2 linear1",     T, 4096, 1024, false},
        {"FF2 linear2",     T, 1024, 4096, false},   // known gap
        // Encoder projection (once)
        {"Enc proj",        T, 640, 1024, false},
        // Decoder GEMVs (per step)
        {"LSTM0 gates",     1, 2560, 1280, true},
        {"LSTM1 gates",     1, 2560, 1280, true},
        {"Dec proj",        1, 640, 640, false},
        {"Out proj (dec)",  1, 1030, 640, false},
        // Subsampling
        {"Sub conv.3 pw",   256, 256*32, 256, false},  // approximate
        {"Sub conv.6 pw",   256, 256*16, 256, false},  // approximate
        {"Sub out linear",  T, 1024, 4096, false},
    };

    // Also test longer audio (T=375 for ~60s)
    int T_long = 375;
    int pos_len_long = 2 * T_long - 1;
    Shape shapes_long[] = {
        {"FF1 linear2 (60s)", T_long, 1024, 4096, false},
        {"Fused QKV (60s)",   T_long, 3072, 1024, false},
        {"Conv pw1 (60s)",    T_long, 2048, 1024, true},
        {"FF1 linear1 (60s)", T_long, 4096, 1024, false},
        {"Out proj (60s)",    T_long, 1024, 1024, false},
        {"Pos proj (60s)",    pos_len_long, 1024, 1024, false},
    };

    printf("\n=== T=%d (~10s audio) — per-block shapes run 24x ===\n\n", T);
    for (auto& s : shapes)
        bench_shape(stream, s, buf, 50, 200);

    printf("\n=== T=%d (~60s audio) ===\n\n", T_long);
    for (auto& s : shapes_long)
        bench_shape(stream, s, buf, 50, 200);

    cudaFree(buf);
    cudaFree(s_workspace);
    cublasDestroy(s_cublas);
    cublasLtDestroy(s_cublaslt);
    cudaStreamDestroy(stream);
    return 0;
}
