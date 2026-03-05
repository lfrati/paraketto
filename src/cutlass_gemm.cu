// cutlass_gemm.cu — CUTLASS FP16 GEMM kernels replacing cuBLAS
//
// Instantiates SM80 TensorOp templates benchmarked against cuBLAS for
// every GEMM shape in parakeet. Configs match or beat cuBLAS on all
// shapes except FF1 linear2 (1024x63x4096) split-K where cuBLAS is 20% faster.
//
// Build: nvcc -std=c++17 -arch=sm_80 -O3 -I../third_party/cutlass/include \
//        --expt-relaxed-constexpr -c cutlass_gemm.cu -o cutlass_gemm.o

#include "cutlass_gemm.h"
#include "kernels.h"  // for residual_add_fp16 (bias add)

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// =========================================================================
// CUTLASS kernel type aliases — SM80 FP16 TensorOp
// =========================================================================

template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// --- NN (ColumnMajor, ColumnMajor in cuBLAS convention) ---

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

using GemmNN_64x64_32_s3 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 3, 8, 8>;

using GemmNN_64x64_32_s6_a2 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<2>, Swizzle, 6, 2, 8>;

using GemmNN_128x128_32_s5 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 5, 8, 8>;

// --- TN (RowMajor A, ColumnMajor B in cuBLAS convention) ---

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

// --- Batched GEMM types (single kernel launch for all batches) ---

using BatchSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// NN batched, full align=8 (for T%8==0: all lds and strides 8-aligned)
using BatchedNN_64x64_64_s6 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, BatchSwizzle, 6, 8, 8>;

// NN batched, mixed: A 8-aligned, B 1-aligned, epilogue 8-aligned, stages=2
// For weighted_sum when T is odd: ldA=HEAD_DIM(64), ldB=T(odd), ldC=HEAD_DIM(64)
using BatchedNN_64x64_32_s2_a8b1 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, BatchSwizzle, 2, 8, 1>;

// TN batched, full align=8 (for T%8==0)
using BatchedTN_64x64_64_s6 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, BatchSwizzle, 6, 8, 8>;

// TN batched, mixed: inputs 8-aligned, epilogue 1-aligned, stages=6
// For content_scores/pos_scores: input lds always 8-aligned (HEAD_DIM=64, D_MODEL=1024)
// but output ld=T or 2T-1 may be odd. Full pipeline speed with scalar epilogue stores.
using BatchedTN_64x64_64_s6_e1 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<1>, BatchSwizzle, 6, 8, 8>;

// =========================================================================
// Workspace (for split-K if needed in the future)
// =========================================================================

static void* s_workspace = nullptr;
static size_t s_workspace_size = 0;

void cutlass_gemm_init(cudaStream_t) {
    // Current configs don't need workspace (no split-K).
    // Reserve 4MB for future split-K or other needs.
    s_workspace_size = 4 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&s_workspace, s_workspace_size));
}

void cutlass_gemm_free() {
    if (s_workspace) { cudaFree(s_workspace); s_workspace = nullptr; }
}

// =========================================================================
// Generic CUTLASS GEMM runner
// =========================================================================

template<typename GemmOp>
static void run_gemm(int M, int N, int K,
                     const half* A, int ldA,
                     const half* B, int ldB,
                     half* C, int ldC,
                     half alpha, half beta,
                     cudaStream_t stream) {
    using ElementA = cutlass::half_t;
    using ElementC = cutlass::half_t;

    typename GemmOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(A), ldA},
        {reinterpret_cast<const ElementA*>(B), ldB},
        {reinterpret_cast<ElementC*>(C), ldC},
        {reinterpret_cast<ElementC*>(C), ldC},
        {*reinterpret_cast<ElementC*>(&alpha), *reinterpret_cast<ElementC*>(&beta)}
    );

    GemmOp op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS can_implement failed for %dx%dx%d: %d\n", M, N, K, (int)status);
        exit(1);
    }

    size_t ws_size = GemmOp::get_workspace_size(args);
    void* ws = (ws_size > 0 && ws_size <= s_workspace_size) ? s_workspace : nullptr;

    status = op.initialize(args, ws, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS initialize failed for %dx%dx%d: %d\n", M, N, K, (int)status);
        exit(1);
    }

    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS run failed for %dx%dx%d: %d\n", M, N, K, (int)status);
        exit(1);
    }
}

// =========================================================================
// Generic CUTLASS batched GEMM runner (single kernel launch for all batches)
// =========================================================================

template<typename BatchedGemmOp>
static void run_batched_gemm(int M, int N, int K,
                              const half* A, int ldA, long long strideA,
                              const half* B, int ldB, long long strideB,
                              half* C, int ldC, long long strideC,
                              int batch,
                              half alpha, half beta,
                              cudaStream_t stream) {
    using ElementA = cutlass::half_t;
    using ElementC = cutlass::half_t;

    typename BatchedGemmOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(A), ldA}, strideA,
        {reinterpret_cast<const ElementA*>(B), ldB}, strideB,
        {reinterpret_cast<const ElementC*>(C), ldC}, strideC,
        {reinterpret_cast<ElementC*>(C), ldC}, strideC,
        {*reinterpret_cast<ElementC*>(&alpha), *reinterpret_cast<ElementC*>(&beta)},
        batch
    );

    BatchedGemmOp op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS batched can_implement failed for %dx%dx%d batch=%d: %d\n",
                M, N, K, batch, (int)status);
        exit(1);
    }

    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS batched initialize failed for %dx%dx%d batch=%d: %d\n",
                M, N, K, batch, (int)status);
        exit(1);
    }

    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS batched run failed for %dx%dx%d batch=%d: %d\n",
                M, N, K, batch, (int)status);
        exit(1);
    }
}

// Helper: pick alignment for batched GEMM (considers leading dims AND strides)
static int pick_batched_align(int ld1, int ld2, int ld3,
                               long long s1, long long s2, long long s3) {
    if ((ld1 % 8 == 0) && (ld2 % 8 == 0) && (ld3 % 8 == 0) &&
        (s1 % 8 == 0) && (s2 % 8 == 0) && (s3 % 8 == 0)) return 8;
    return 1;
}

// =========================================================================
// NN dispatch: Row-major Y[m,n] = X[m,k] @ W[k,n]
//   → cuBLAS convention: col_m=n, col_n=m, col_k=k, A=W, B=X
// =========================================================================

static void nn_dispatch(cudaStream_t stream,
                        const half* X, int m, int k,
                        const half* W, int n, half* Y,
                        half alpha, half beta) {
    // Column-major params for CUTLASS
    int col_m = n, col_n = m, col_k = k;
    int ldA = n, ldB = k, ldC = n;

    // Shape-based config selection (from benchmark results):
    //   3072x63x1024 (fused QKV)  → 64x64_32_s10  (22% faster)
    //   1008x256x256 (sub conv.6) → 64x64_32_s10  (36% faster)
    //   640x1x640    (dec proj)   → 64x64_32_s3   (33% faster)
    //   4000x256x256 (sub conv.3) → 128x128_32_s5
    //   1030x1x640   (out proj)   → 64x64_32_s6_a2 (align2 for non-8-aligned M)
    //   Everything else            → 64x64_64_s6   (default, matched cuBLAS)

    if (col_m % 8 != 0) {
        // Non-8-aligned M (out_proj 1030x1x640): align2 config
        run_gemm<GemmNN_64x64_32_s6_a2>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
    } else if (col_m == 3072 || (col_m <= 1024 && col_n >= 128 && col_k == 256)) {
        // Fused QKV (3072x63x1024) and sub conv.6 (1008x256x256): deep pipeline
        run_gemm<GemmNN_64x64_32_s10>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
    } else if (col_n == 1 && col_k <= 640) {
        // Decoder GEMV (dec_proj 640x1x640): small tile, 3 stages
        run_gemm<GemmNN_64x64_32_s3>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
    } else if (col_m >= 2048 && col_n >= 128) {
        // Large subsampling (4000x256x256): big tiles
        run_gemm<GemmNN_128x128_32_s5>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
    } else {
        // Default: 64x64 tile, K=64, 6 stages (matched cuBLAS everywhere else)
        run_gemm<GemmNN_64x64_64_s6>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
    }
}

// =========================================================================
// TN dispatch: Row-major Y[m,n] = X[m,k] @ W[n,k]^T
//   → cuBLAS convention: opA=T, opB=N, A=W (stored [k,n]), B=X
// =========================================================================

static void tn_dispatch(cudaStream_t stream,
                        const half* X, int m, int k,
                        const half* W, int n, half* Y,
                        half alpha, half beta) {
    int col_m = n, col_n = m, col_k = k;
    int ldA = k, ldB = k, ldC = n;

    // All TN shapes use 64x64_64_s6 (matched or beat cuBLAS)
    run_gemm<GemmTN_64x64_64_s6>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
}

// =========================================================================
// Public API
// =========================================================================

static const half HALF_ONE = __float2half(1.0f);
static const half HALF_ZERO = __float2half(0.0f);

void cutlass_gemm_nn(cudaStream_t stream,
                     const half* X, int m, int k,
                     const half* W, int n, half* Y) {
    nn_dispatch(stream, X, m, k, W, n, Y, HALF_ONE, HALF_ZERO);
}

void cutlass_gemm_nn_bias(cudaStream_t stream,
                          const half* X, int m, int k,
                          const half* W, int n,
                          const half* bias, half* Y) {
    nn_dispatch(stream, X, m, k, W, n, Y, HALF_ONE, HALF_ZERO);
    bias_add_fp16(Y, bias, m, n, stream);
}

void cutlass_gemm_nt(cudaStream_t stream,
                     const half* X, int m, int k,
                     const half* W, int n, half* Y) {
    tn_dispatch(stream, X, m, k, W, n, Y, HALF_ONE, HALF_ZERO);
}

void cutlass_gemm_nt_accum(cudaStream_t stream,
                           const half* X, int m, int k,
                           const half* W, int n, half* Y) {
    tn_dispatch(stream, X, m, k, W, n, Y, HALF_ONE, HALF_ONE);
}

void cutlass_gemm_nt_bias(cudaStream_t stream,
                          const half* X, int m, int k,
                          const half* W, int n,
                          const half* bias, half* Y) {
    tn_dispatch(stream, X, m, k, W, n, Y, HALF_ONE, HALF_ZERO);
    bias_add_fp16(Y, bias, m, n, stream);
}

void cutlass_gemm_nt_accum_bias(cudaStream_t stream,
                                const half* X, int m, int k,
                                const half* W, int n,
                                const half* bias, half* Y) {
    tn_dispatch(stream, X, m, k, W, n, Y, HALF_ONE, HALF_ONE);
    bias_add_fp16(Y, bias, m, n, stream);
}

// =========================================================================
// Batched strided GEMM (single kernel launch via GemmBatched)
// =========================================================================


void cutlass_batched_gemm_nn(cudaStream_t stream,
                             const half* A, const half* B, half* C,
                             int batch, int m, int n, int k,
                             long long strideA, long long strideB, long long strideC) {
    // Row-major NN: C[b,m,n] = A[b,m,k] @ B[b,k,n]
    // → CUTLASS col-major NN: col_m=n, col_n=m, A_c=B, B_c=A
    int col_m = n, col_n = m, col_k = k;
    int ldA_c = n, ldB_c = k, ldC_c = n;
    int align = pick_batched_align(ldA_c, ldB_c, ldC_c, strideB, strideA, strideC);
    if (align == 8)
        run_batched_gemm<BatchedNN_64x64_64_s6>(col_m, col_n, col_k,
            B, ldA_c, strideB, A, ldB_c, strideA, C, ldC_c, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
    else
        // Mixed: A(ldA_c=HEAD_DIM=64) always 8-aligned, C(ldC_c=64) 8-aligned
        // Only B(ldB_c=T) may be unaligned → stages=2, AlignmentA=8, AlignmentB=1
        run_batched_gemm<BatchedNN_64x64_32_s2_a8b1>(col_m, col_n, col_k,
            B, ldA_c, strideB, A, ldB_c, strideA, C, ldC_c, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
}

void cutlass_batched_gemm_nt(cudaStream_t stream,
                             const half* A, const half* B, half* C,
                             int batch, int m, int n, int k,
                             long long strideA, long long strideB, long long strideC) {
    // Row-major NT: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
    // → CUTLASS col-major TN: col_m=n, col_n=m, A_c=B(RowMajor ld=k), B_c=A(ColMajor ld=k)
    int col_m = n, col_n = m, col_k = k;
    int ldA_c = k, ldB_c = k, ldC_c = n;
    int align = pick_batched_align(ldA_c, ldB_c, ldC_c, strideB, strideA, strideC);
    if (align == 8)
        run_batched_gemm<BatchedTN_64x64_64_s6>(col_m, col_n, col_k,
            B, ldA_c, strideB, A, ldB_c, strideA, C, ldC_c, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
    else
        // Mixed: inputs always 8-aligned (ld=HEAD_DIM=64), only output ld=T may be odd
        // → full pipeline stages=6 with scalar epilogue
        run_batched_gemm<BatchedTN_64x64_64_s6_e1>(col_m, col_n, col_k,
            B, ldA_c, strideB, A, ldB_c, strideA, C, ldC_c, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
}

void cutlass_batched_gemm_nn_ex(cudaStream_t stream,
                                const half* A, int ldA, long long strideA,
                                const half* B, int ldB, long long strideB,
                                half* C, int ldC, long long strideC,
                                int batch, int m, int n, int k) {
    // Row-major NN with explicit ld/stride:
    // → CUTLASS col-major NN: col_m=n, col_n=m, A_c=B(ldB), B_c=A(ldA)
    int col_m = n, col_n = m, col_k = k;
    int align = pick_batched_align(ldB, ldA, ldC, strideB, strideA, strideC);
    if (align == 8)
        run_batched_gemm<BatchedNN_64x64_64_s6>(col_m, col_n, col_k,
            B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
    else
        run_batched_gemm<BatchedNN_64x64_32_s2_a8b1>(col_m, col_n, col_k,
            B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
}

void cutlass_batched_gemm_nt_ex(cudaStream_t stream,
                                const half* A, int ldA, long long strideA,
                                const half* B, int ldB, long long strideB,
                                half* C, int ldC, long long strideC,
                                int batch, int m, int n, int k) {
    // Row-major NT with explicit ld/stride:
    // → CUTLASS col-major TN: col_m=n, col_n=m, A_c=B(ldB), B_c=A(ldA)
    int col_m = n, col_n = m, col_k = k;
    int align = pick_batched_align(ldB, ldA, ldC, strideB, strideA, strideC);
    if (align == 8)
        run_batched_gemm<BatchedTN_64x64_64_s6>(col_m, col_n, col_k,
            B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
    else
        run_batched_gemm<BatchedTN_64x64_64_s6_e1>(col_m, col_n, col_k,
            B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
            batch, HALF_ONE, HALF_ZERO, stream);
}

// =========================================================================
// gemm.h unified interface — forward to cutlass_* functions
// =========================================================================

#include "gemm.h"

void gemm_init(cudaStream_t stream)   { cutlass_gemm_init(stream); }
void gemm_free()                      { cutlass_gemm_free(); }

void gemm_nn(cudaStream_t s, const half* X, int m, int k, const half* W, int n, half* Y)
    { cutlass_gemm_nn(s, X, m, k, W, n, Y); }
void gemm_nn_bias(cudaStream_t s, const half* X, int m, int k, const half* W, int n, const half* bias, half* Y)
    { cutlass_gemm_nn_bias(s, X, m, k, W, n, bias, Y); }
void gemm_nt(cudaStream_t s, const half* X, int m, int k, const half* W, int n, half* Y)
    { cutlass_gemm_nt(s, X, m, k, W, n, Y); }
void gemm_nt_bias(cudaStream_t s, const half* X, int m, int k, const half* W, int n, const half* bias, half* Y)
    { cutlass_gemm_nt_bias(s, X, m, k, W, n, bias, Y); }

void batched_gemm_nn(cudaStream_t s, const half* A, const half* B, half* C,
                     int batch, int m, int n, int k,
                     long long sA, long long sB, long long sC)
    { cutlass_batched_gemm_nn(s, A, B, C, batch, m, n, k, sA, sB, sC); }
void batched_gemm_nt(cudaStream_t s, const half* A, const half* B, half* C,
                     int batch, int m, int n, int k,
                     long long sA, long long sB, long long sC)
    { cutlass_batched_gemm_nt(s, A, B, C, batch, m, n, k, sA, sB, sC); }
void batched_gemm_nt_ex(cudaStream_t s,
                        const half* A, int ldA, long long sA,
                        const half* B, int ldB, long long sB,
                        half* C, int ldC, long long sC,
                        int batch, int m, int n, int k)
    { cutlass_batched_gemm_nt_ex(s, A, ldA, sA, B, ldB, sB, C, ldC, sC, batch, m, n, k); }
