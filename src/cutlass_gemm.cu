// cutlass_gemm.cu — CUTLASS FP16 GEMM kernels replacing cuBLAS
//
// Instantiates SM80 TensorOp templates benchmarked against cuBLAS for
// every GEMM shape in parakeet. Configs match or beat cuBLAS on all
// shapes except: FF linear2 T≤64 (-20%), T=257-279 (-8.5%, cuBLAS uses
// 256x64 split-K with GemmIdentitySwizzle which CUTLASS 2.x can't replicate).
//
// Build: nvcc -std=c++17 -arch=sm_80 -O3 -I../third_party/cutlass/include \
//        --expt-relaxed-constexpr -c cutlass_gemm.cu -o cutlass_gemm.o

#include "cutlass_gemm.h"
#include "kernels.h"  // for residual_add_fp16 (bias add)

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

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

// --- NN split-K parallel (for thin-N, large-K shapes like FF linear2) ---

using GemmSplitKNN_64x64_64_s6 = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>>;

// 128x64 split-K (T=65-128: 12-14% faster than cuBLAS)
using GemmSplitKNN_128x64_32_s5 = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>>;

// --- NN tiles for FF linear2 at medium T ---

using GemmNN_128x64_k32_s5 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 5, 8, 8>;

// 64x128 tile (T=241-448: tied or better than 128x64)
using GemmNN_64x128_k32_s5 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 5, 8, 8>;

// 128x128 split-K (T=449-800: beats cuBLAS by 7-19% with sk=2/3/4)
using GemmSplitKNN_128x128_32_s5 = cutlass::gemm::device::GemmSplitKParallel<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>>;

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
    // Split-K workspace: 128x128 tile, sk=4, T=800: 1024*800*2*4 = 6.5MB. 32MB is safe.
    s_workspace_size = 32 * 1024 * 1024;
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
// Generic CUTLASS split-K parallel GEMM runner
// =========================================================================

template<typename SplitKGemmOp>
static void run_splitk_gemm(int M, int N, int K,
                             const half* A, int ldA,
                             const half* B, int ldB,
                             half* C, int ldC,
                             int split_k_slices,
                             half alpha, half beta,
                             cudaStream_t stream) {
    using ElementA = cutlass::half_t;
    using ElementC = cutlass::half_t;

    typename SplitKGemmOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(A), ldA},
        {reinterpret_cast<const ElementA*>(B), ldB},
        {reinterpret_cast<ElementC*>(C), ldC},
        {reinterpret_cast<ElementC*>(C), ldC},
        {*reinterpret_cast<ElementC*>(&alpha), *reinterpret_cast<ElementC*>(&beta)},
        split_k_slices
    );

    SplitKGemmOp op;
    size_t ws_size = SplitKGemmOp::get_workspace_size(args);
    void* ws = (ws_size > 0 && ws_size <= s_workspace_size) ? s_workspace : nullptr;

    auto status = op.initialize(args, ws);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS split-K initialize failed for %dx%dx%d sk=%d: %d\n",
                M, N, K, split_k_slices, (int)status);
        exit(1);
    }

    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS split-K run failed for %dx%dx%d sk=%d: %d\n",
                M, N, K, split_k_slices, (int)status);
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
    //   FF linear2 (1024×T×4096, 48 calls/utterance):
    //     T≤64   → 64x64_sk4   (-20% vs cuBLAS; cuBLAS wins here)
    //     T≤128  → 128x64_sk4  (-14% vs cuBLAS; actually BEATS cuBLAS)
    //     T≤192  → 64x64_s6    (-11% vs cuBLAS; beats)
    //     T≤256  → 128x64_sk4  (ties cuBLAS at 20.5us)
    //     T≤448  → 64x128_s5   (ties cuBLAS T≥280; -8.5% at T=257-279)
    //     T≤512  → 128x128_sk2 (beats cuBLAS by 7%)
    //     T≤640  → 128x128_sk3 (beats cuBLAS by 9%)
    //     T≤800  → 128x128_sk4 (beats cuBLAS by 12-19%)
    //     T>800  → 128x128_s5  (matched cuBLAS)
    //   Everything else → 64x64_64_s6 (default, matched cuBLAS)

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
    } else if (col_k >= 4096 && col_n <= 64) {
        // FF linear2 T≤64: split-K sk=4 with 64x64 tile (-20% vs cuBLAS)
        run_splitk_gemm<GemmSplitKNN_64x64_64_s6>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, 4, alpha, beta, stream);
    } else if (col_k >= 4096 && col_n <= 128) {
        // FF linear2 T=65-128: split-K sk=4 with 128x64 tile (-14% vs cuBLAS)
        run_splitk_gemm<GemmSplitKNN_128x64_32_s5>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, 4, alpha, beta, stream);
    } else if (col_k >= 4096 && col_n <= 192) {
        // FF linear2 T=129-192: 64x64 default (-11% vs cuBLAS)
        run_gemm<GemmNN_64x64_64_s6>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
    } else if (col_k >= 4096 && col_n <= 256) {
        // FF linear2 T=193-256: 128x64 split-K sk=4 (ties cuBLAS at 20.5us)
        run_splitk_gemm<GemmSplitKNN_128x64_32_s5>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, 4, alpha, beta, stream);
    } else if (col_k >= 4096 && col_n <= 448) {
        // FF linear2 T=257-448: 64x128 tile (ties cuBLAS at T>=280, 8.5% slower T=257-279)
        run_gemm<GemmNN_64x128_k32_s5>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, alpha, beta, stream);
    } else if (col_k >= 4096 && col_n <= 512) {
        // FF linear2 T=449-512: 128x128 sk=2 (-7% vs cuBLAS)
        run_splitk_gemm<GemmSplitKNN_128x128_32_s5>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, 2, alpha, beta, stream);
    } else if (col_k >= 4096 && col_n <= 640) {
        // FF linear2 T=513-640: 128x128 sk=3 (-9% vs cuBLAS)
        run_splitk_gemm<GemmSplitKNN_128x128_32_s5>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, 3, alpha, beta, stream);
    } else if (col_k >= 4096 && col_n <= 800) {
        // FF linear2 T=641-800: 128x128 sk=4 (-12-19% vs cuBLAS)
        run_splitk_gemm<GemmSplitKNN_128x128_32_s5>(col_m, col_n, col_k, W, ldA, X, ldB, Y, ldC, 4, alpha, beta, stream);
    } else if (col_k >= 4096) {
        // FF linear2 T>800: 128x128 regular (matched cuBLAS)
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
