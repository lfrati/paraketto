// cutlass_fp8_gemm.cu — CUTLASS 3.x SM120 FP8 E4M3 GEMM kernels
//
// Single kernel instantiation following CUTLASS example 87a (Blackwell GeForce):
//   LayoutA = RowMajor, LayoutB = ColumnMajor
//
// SM120 blockwise scaling MMA uses stride_B = (K, 1, 0), meaning B is read as
// [N,K] RowMajor despite declaring ColumnMajor. The caller must provide B as
// [N,K] row-major (K contiguous per row of N).
//
// For NN GEMMs (W stored as [K,N] row-major): transpose to [N,K] before calling.
// For NT GEMMs (W stored as [N,K] row-major): pass directly.
//
// Per-tensor scaling via trivial blockwise: all SFA/SFB = 1.0f, alpha = act_scale * wt_scale.
//
// Build: nvcc -std=c++17 -O3 -arch=sm_120a $(CUTLASS_INC) -c cutlass_fp8_gemm.cu

#include "cutlass_fp8_gemm.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

using namespace cute;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// =========================================================================
// CUTLASS 3.x SM120 FP8 E4M3 kernel type definitions
// =========================================================================

// Element types
using ElementA   = cutlass::float_e4m3_t;
using ElementB   = cutlass::float_e4m3_t;
using ElementC   = cutlass::half_t;
using ElementD   = cutlass::half_t;
using ElementAcc = float;
using ElementCmp = float;

// Layouts — SM120 TN: A=RowMajor, B=ColumnMajor (matching example 87a)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Alignments: 128-bit memory transactions
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;   // 16
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;   // 16
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;   // 8
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;   // 8

// Tile and cluster shapes (GeForce: no multicast TMA, cluster must be 1x1x1)
using TileShape    = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// Trivial blockwise scale config: SFVecSize = TileSize, all scales = 1.0f
using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(TileShape{}));
using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

// Epilogue
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCmp,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop: A=RowMajor, B=ColumnMajor (TN)
// Computes D[M,N] = alpha * A[M,K] @ B[N,K]^T
// where B[N,K] ColumnMajor = W[K,N] row-major in memory
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// =========================================================================
// Module state — pre-allocated scale buffers and workspace
// =========================================================================

static float*  s_sfa = nullptr;     // SFA buffer, all 1.0f
static float*  s_sfb = nullptr;     // SFB buffer, all 1.0f
static void*   s_workspace = nullptr;
static size_t  s_workspace_size = 0;

// =========================================================================
// Init / Free
// =========================================================================

void cutlass_fp8_init(int max_M, int max_N, int max_K, cudaStream_t stream) {
    // Pad to scale granularity (128) — same as run_fp8_gemm
    constexpr int SCALE_GRAN_M = 128;
    max_M = (max_M + SCALE_GRAN_M - 1) / SCALE_GRAN_M * SCALE_GRAN_M;

    // Compute scale buffer sizes for the maximum problem shape.
    // Over-allocate to 256KB each — the trivial blockwise scale layout may use
    // sparse strides that access offsets beyond size(filter_zeros(layout)) when
    // called with smaller problem shapes. Filling a large contiguous buffer with
    // 1.0f ensures any access pattern reads 1.0f regardless of layout strides.
    constexpr size_t SF_BUF_ELEMS = 64 * 1024;  // 256KB per buffer
    size_t sfa_elems = SF_BUF_ELEMS;
    size_t sfb_elems = SF_BUF_ELEMS;

    CUDA_CHECK(cudaMalloc(&s_sfa, sfa_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s_sfb, sfb_elems * sizeof(float)));

    // Fill with 1.0f (trivial per-tensor scaling)
    std::vector<float> ones_a(sfa_elems, 1.0f);
    std::vector<float> ones_b(sfb_elems, 1.0f);
    CUDA_CHECK(cudaMemcpyAsync(s_sfa, ones_a.data(), sfa_elems * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(s_sfb, ones_b.data(), sfb_elems * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    // Query workspace size at max shape
    s_workspace_size = 0;
    {
        auto ws_layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(max_M, max_N, max_K, 1));
        auto ws_layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(max_M, max_N, max_K, 1));
        using SA = typename Gemm::GemmKernel::StrideA;
        using SB = typename Gemm::GemmKernel::StrideB;
        using SC = typename Gemm::GemmKernel::StrideC;
        using SD = typename Gemm::GemmKernel::StrideD;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {max_M, max_N, max_K, 1},
            {nullptr, cutlass::make_cute_packed_stride(SA{}, make_shape(max_M, max_K, 1)),
             nullptr, cutlass::make_cute_packed_stride(SB{}, make_shape(max_N, max_K, 1)),
             s_sfa, ws_layout_sfa, s_sfb, ws_layout_sfb},
            {{}, nullptr, cutlass::make_cute_packed_stride(SC{}, make_shape(max_M, max_N, 1)),
                 nullptr, cutlass::make_cute_packed_stride(SD{}, make_shape(max_M, max_N, 1))}
        };
        args.epilogue.thread.alpha = 1.0f;
        args.epilogue.thread.beta = 0.0f;
        s_workspace_size = Gemm::get_workspace_size(args);
    }
    if (s_workspace_size > 0)
        CUDA_CHECK(cudaMalloc(&s_workspace, s_workspace_size));
}

void cutlass_fp8_free() {
    if (s_sfa) { cudaFree(s_sfa); s_sfa = nullptr; }
    if (s_sfb) { cudaFree(s_sfb); s_sfb = nullptr; }
    if (s_workspace) { cudaFree(s_workspace); s_workspace = nullptr; }
}

// =========================================================================
// GEMM runner
//
// Computes: D[M,N] = alpha * A[M,K] @ B^T
//   A: [M,K] row-major FP8 (activation)
//   B: [N,K] row-major FP8 (weight, pre-transposed by caller)
//   D: [M,N] row-major FP16 (output)
// =========================================================================

static int run_fp8_gemm(cudaStream_t stream, int M, int N, int K,
                         const uint8_t* A, const uint8_t* B, half* C, float alpha) {
    // SM120 blockwise scaling requires M to be a multiple of the scale granularity (128).
    constexpr int SCALE_GRAN_M = 128;
    int M_padded = (M + SCALE_GRAN_M - 1) / SCALE_GRAN_M * SCALE_GRAN_M;

    using SA = typename Gemm::GemmKernel::StrideA;
    using SB = typename Gemm::GemmKernel::StrideB;
    using SC = typename Gemm::GemmKernel::StrideC;
    using SD = typename Gemm::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(SA{}, make_shape(M_padded, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(SB{}, make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(SC{}, make_shape(M_padded, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(SD{}, make_shape(M_padded, N, 1));
    auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M_padded, N, K, 1));
    auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M_padded, N, K, 1));

    // Use pre-allocated scale buffers (256KB each, all 1.0f from cutlass_fp8_init).
    // The trivial blockwise layout strides vary by problem shape but always index
    // within the 256KB buffer — filled uniformly with 1.0f so any access reads 1.0f.
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M_padded, N, K, 1},
        {reinterpret_cast<const ElementA*>(A), stride_A,
         reinterpret_cast<const ElementB*>(B), stride_B,
         s_sfa, layout_sfa,
         s_sfb, layout_sfb},
        {{}, reinterpret_cast<const ElementC*>(C), stride_C,
             reinterpret_cast<ElementD*>(C), stride_D}
    };
    args.epilogue.thread.alpha = alpha;
    args.epilogue.thread.beta = 0.0f;

    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 can_implement failed: M=%d N=%d K=%d status=%d\n",
                M, N, K, (int)status);
        return -1;
    }
    status = gemm.initialize(args, s_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 initialize failed: M=%d N=%d K=%d status=%d\n",
                M, N, K, (int)status);
        return -2;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS FP8 run failed: M=%d N=%d K=%d status=%d\n",
                M, N, K, (int)status);
        return -3;
    }
    return 0;
}

// =========================================================================
// Public API
//
// B must be [N,K] row-major (K contiguous). The caller is responsible for
// transposing NN weights from [K,N] to [N,K] before calling.
// NT weights are naturally [N,K] and can be passed directly.
// =========================================================================

int cutlass_fp8_nn(cudaStream_t stream, int M, int N, int K,
                   const uint8_t* A, const uint8_t* B, half* C, float alpha) {
    return run_fp8_gemm(stream, M, N, K, A, B, C, alpha);
}
