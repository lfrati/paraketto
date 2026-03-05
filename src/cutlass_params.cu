// cutlass_params.cu — Host-side CUTLASS Params construction for cudaless launch
//
// Compiled with nvcc to handle CUTLASS templates. Produces an object file
// that constructs Params structs WITHOUT calling any CUDA runtime functions.
// The cudaless binary links this object but never calls cudaMalloc/cudaFree etc.
//
// Usage: call cutlass_*_build_params() to fill a raw byte buffer with the
//        kernel's Params struct, then pass it as kernel args in cbuf0.

// Stub out CUDA runtime registration to avoid linking libcudart.
// nvcc emits these calls for .cu files, but we never launch device code.
// Stubs defined AFTER nvcc includes, via linker override.


#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

#include <cstdint>
#include <cstring>

// =========================================================================
// Same type aliases as cutlass_gemm.cu
// =========================================================================

template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using BatchSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// NN variants
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

// TN variant
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

// Batched variants
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
// Generic Gemm params builder
// =========================================================================

template<typename GemmOp>
static int build_gemm_params_impl(void* out_params, int out_size,
                                   int* out_grid, int* out_block, int* out_smem,
                                   int M, int N, int K,
                                   uint64_t ptr_A, int ldA,
                                   uint64_t ptr_B, int ldB,
                                   uint64_t ptr_C, int ldC,
                                   uint16_t alpha_fp16, uint16_t beta_fp16) {
    using ElementA = cutlass::half_t;
    using ElementC = cutlass::half_t;

    ElementC alpha_c, beta_c;
    memcpy(&alpha_c, &alpha_fp16, 2);
    memcpy(&beta_c, &beta_fp16, 2);

    typename GemmOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(ptr_A), ldA},
        {reinterpret_cast<const ElementA*>(ptr_B), ldB},
        {reinterpret_cast<ElementC*>(ptr_C), ldC},
        {reinterpret_cast<ElementC*>(ptr_C), ldC},
        {alpha_c, beta_c}
    );

    GemmOp op;
    memset(&op, 0, sizeof(op));
    auto status = op.initialize(args, nullptr, nullptr);
    if (status != cutlass::Status::kSuccess) return -1;

    // Copy params bytes
    using Params = typename GemmOp::GemmKernel::Params;
    int params_size = sizeof(Params);
    if (params_size > out_size) return -1;
    memcpy(out_params, &op, params_size);

    // Extract grid from the internal Params struct (grid_tiled_shape at offset 12-23)
    // CUTLASS initialize() may rearrange problem dimensions internally, so the
    // internal grid_tiled_shape can differ from a naive external computation.
    // The kernel expects the QMD grid to match its internal grid_tiled_shape.
    const int32_t* params_words = (const int32_t*)out_params;
    out_grid[0] = params_words[3];  // grid_tiled_shape.m
    out_grid[1] = params_words[4];  // grid_tiled_shape.n
    out_grid[2] = params_words[5];  // grid_tiled_shape.k (split_k, usually 1)

    using KernelType = typename GemmOp::GemmKernel;
    out_block[0] = KernelType::kThreadCount;
    out_block[1] = 1;
    out_block[2] = 1;
    *out_smem = (int)sizeof(typename KernelType::SharedStorage);

    return params_size;
}

template<typename BatchedGemmOp>
static int build_batched_params_impl(void* out_params, int out_size,
                                      int* out_grid, int* out_block, int* out_smem,
                                      int M, int N, int K,
                                      uint64_t ptr_A, int ldA, long long strideA,
                                      uint64_t ptr_B, int ldB, long long strideB,
                                      uint64_t ptr_C, int ldC, long long strideC,
                                      int batch,
                                      uint16_t alpha_fp16, uint16_t beta_fp16) {
    using ElementA = cutlass::half_t;
    using ElementC = cutlass::half_t;

    ElementC alpha_c, beta_c;
    memcpy(&alpha_c, &alpha_fp16, 2);
    memcpy(&beta_c, &beta_fp16, 2);

    typename BatchedGemmOp::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(ptr_A), ldA}, strideA,
        {reinterpret_cast<const ElementA*>(ptr_B), ldB}, strideB,
        {reinterpret_cast<const ElementC*>(ptr_C), ldC}, strideC,
        {reinterpret_cast<ElementC*>(ptr_C), ldC}, strideC,
        {alpha_c, beta_c},
        batch
    );

    BatchedGemmOp op;
    memset(&op, 0, sizeof(op));
    auto status = op.initialize(args, nullptr, nullptr);
    if (status != cutlass::Status::kSuccess) return -1;

    using Params = typename BatchedGemmOp::GemmKernel::Params;
    int params_size = sizeof(Params);
    if (params_size > out_size) return -1;
    memcpy(out_params, &op, params_size);

    // Extract grid from internal Params (grid_tiled_shape at offset 12-23)
    const int32_t* params_words = (const int32_t*)out_params;
    out_grid[0] = params_words[3];  // grid_tiled_shape.m
    out_grid[1] = params_words[4];  // grid_tiled_shape.n
    out_grid[2] = params_words[5];  // grid_tiled_shape.k (= batch for GemmBatched)

    using KernelType = typename BatchedGemmOp::GemmKernel;
    out_block[0] = KernelType::kThreadCount;
    out_block[1] = 1;
    out_block[2] = 1;
    *out_smem = (int)sizeof(typename KernelType::SharedStorage);

    return params_size;
}

// =========================================================================
// Public C API — one function per GEMM variant
// =========================================================================

extern "C" {

// Returns params_size, fills out_params/out_grid/out_block/out_smem. -1 on error.
// All GEMM functions take CUTLASS col-major convention: M, N, K, ldA, ldB, ldC.

#define GEMM_SIG(name) \
int name(void* out, int out_sz, int* grid, int* block, int* smem, \
    int M, int N, int K, uint64_t A, int ldA, uint64_t B, int ldB, \
    uint64_t C, int ldC, uint16_t alpha, uint16_t beta)

#define BATCH_SIG(name) \
int name(void* out, int out_sz, int* grid, int* block, int* smem, \
    int M, int N, int K, uint64_t A, int ldA, long long sA, \
    uint64_t B, int ldB, long long sB, uint64_t C, int ldC, long long sC, \
    int batch, uint16_t alpha, uint16_t beta)

GEMM_SIG(cutlass_params_nn_64x64_64_s6) {
    return build_gemm_params_impl<GemmNN_64x64_64_s6>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, B, ldB, C, ldC, alpha, beta);
}
GEMM_SIG(cutlass_params_nn_64x64_32_s10) {
    return build_gemm_params_impl<GemmNN_64x64_32_s10>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, B, ldB, C, ldC, alpha, beta);
}
GEMM_SIG(cutlass_params_nn_64x64_32_s3) {
    return build_gemm_params_impl<GemmNN_64x64_32_s3>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, B, ldB, C, ldC, alpha, beta);
}
GEMM_SIG(cutlass_params_nn_64x64_32_s6_a2) {
    return build_gemm_params_impl<GemmNN_64x64_32_s6_a2>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, B, ldB, C, ldC, alpha, beta);
}
GEMM_SIG(cutlass_params_nn_128x128_32_s5) {
    return build_gemm_params_impl<GemmNN_128x128_32_s5>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, B, ldB, C, ldC, alpha, beta);
}
GEMM_SIG(cutlass_params_tn_64x64_64_s6) {
    return build_gemm_params_impl<GemmTN_64x64_64_s6>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, B, ldB, C, ldC, alpha, beta);
}
BATCH_SIG(cutlass_params_batched_nn_64x64_64_s6) {
    return build_batched_params_impl<BatchedNN_64x64_64_s6>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, sA, B, ldB, sB, C, ldC, sC,
        batch, alpha, beta);
}
BATCH_SIG(cutlass_params_batched_nn_64x64_32_s2_a8b1) {
    return build_batched_params_impl<BatchedNN_64x64_32_s2_a8b1>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, sA, B, ldB, sB, C, ldC, sC,
        batch, alpha, beta);
}
BATCH_SIG(cutlass_params_batched_tn_64x64_64_s6) {
    return build_batched_params_impl<BatchedTN_64x64_64_s6>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, sA, B, ldB, sB, C, ldC, sC,
        batch, alpha, beta);
}
BATCH_SIG(cutlass_params_batched_tn_64x64_64_s6_e1) {
    return build_batched_params_impl<BatchedTN_64x64_64_s6_e1>(
        out, out_sz, grid, block, smem, M, N, K, A, ldA, sA, B, ldB, sB, C, ldC, sC,
        batch, alpha, beta);
}

#undef GEMM_SIG
#undef BATCH_SIG

} // extern "C"
