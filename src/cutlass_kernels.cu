// cutlass_kernels.cu — Fused GEMM+activation kernels using CUTLASS
//
// Uses CUTLASS 2.x device API for Sm80 (Ampere) tensor core GEMMs
// with custom epilogues. These kernels are forward-compatible with
// all GPUs >= compute 8.0 including Blackwell (12.0).

#include "cutlass_kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

using ElementA     = cutlass::half_t;
using ElementB     = cutlass::half_t;
using ElementC     = cutlass::half_t;
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int Alignment = 128 / cutlass::sizeof_bits<ElementA>::value; // 8

// ---------------------------------------------------------------------------
// GEMM + SiLU: Y = SiLU(X @ W)
//   X: [M, K] row-major
//   W: [K, N] row-major (ONNX MatMul convention)
//   Y: [M, N] row-major
// ---------------------------------------------------------------------------

using EpilogueSiLU = cutlass::epilogue::thread::LinearCombinationSilu<
    ElementC,       // output type
    Alignment,      // vector width
    ElementAccum,   // accumulator type
    ElementAccum    // compute type
>;

using GemmSiLU = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,   // threadblock
    cutlass::gemm::GemmShape<64, 64, 32>,      // warp
    cutlass::gemm::GemmShape<16, 8, 16>,       // MMA instruction
    EpilogueSiLU,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,          // stages
    Alignment,  // alignment A
    Alignment   // alignment B
>;

// ---------------------------------------------------------------------------
// GEMM + SiLU: Y = SiLU(X @ W^T)
//   X: [M, K] row-major
//   W: [N, K] row-major (Conv/PyTorch convention, transposed)
//   Y: [M, N] row-major
// ---------------------------------------------------------------------------

using LayoutBT = cutlass::layout::ColumnMajor;  // [N, K] row-major = [K, N] col-major

using GemmSiLU_NT = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutBT,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueSiLU,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    Alignment,
    Alignment
>;

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

void gemm_silu_nn_fp16(const half* X, int M, int K,
                        const half* W, int N, half* Y,
                        cudaStream_t stream) {
    GemmSiLU::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(X), K},  // A: [M, K] with lda=K
        {reinterpret_cast<const ElementB*>(W), N},   // B: [K, N] with ldb=N
        {reinterpret_cast<const ElementC*>(Y), N},   // C (unused, beta=0)
        {reinterpret_cast<ElementC*>(Y), N},          // D: [M, N] with ldd=N
        {ElementAccum(1.0f), ElementAccum(0.0f)}     // alpha=1, beta=0
    );

    GemmSiLU gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM+SiLU NN cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return;
    }

    status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM+SiLU NN init failed\n");
        return;
    }

    status = gemm(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM+SiLU NN run failed\n");
    }
}

void gemm_silu_nt_fp16(const half* X, int M, int K,
                        const half* W, int N, half* Y,
                        cudaStream_t stream) {
    GemmSiLU_NT::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(X), K},  // A: [M, K] with lda=K
        {reinterpret_cast<const ElementB*>(W), K},   // B: [N, K] col-major with ldb=K
        {reinterpret_cast<const ElementC*>(Y), N},
        {reinterpret_cast<ElementC*>(Y), N},
        {ElementAccum(1.0f), ElementAccum(0.0f)}
    );

    GemmSiLU_NT gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM+SiLU NT cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return;
    }

    status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM+SiLU NT init failed\n");
        return;
    }

    status = gemm(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM+SiLU NT run failed\n");
    }
}
