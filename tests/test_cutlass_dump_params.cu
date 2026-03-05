// Dump the ACTUAL params that CUTLASS::initialize() produces internally
// Compare against what cutlass_params bridge produces
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using GemmNN = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 6, 8, 8>;

// Our bridge function
extern "C" int cutlass_params_nn_64x64_64_s6(
    void* out, int out_sz, int* grid, int* block, int* smem,
    int M, int N, int K, uint64_t A, int ldA, uint64_t B, int ldB,
    uint64_t C, int ldC, uint16_t alpha, uint16_t beta);

int main() {
    const int M = 64, N = 64, K = 64;
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * 2);
    cudaMalloc(&d_B, K * N * 2);
    cudaMalloc(&d_C, M * N * 2);

    printf("Pointers: A=%p B=%p C=%p\n", d_A, d_B, d_C);

    // Method 1: CUTLASS internal path (what cutlass_gemm.cu does)
    // Row→Col: CUTLASS M=N=64, N=M=64, K=64, A=B_ptr(ldA=64), B=A_ptr(ldB=64)
    using ElementA = cutlass::half_t;
    using ElementC = cutlass::half_t;
    GemmNN::Arguments args(
        {N, M, K},
        {reinterpret_cast<const ElementA*>(d_B), N},
        {reinterpret_cast<const ElementA*>(d_A), K},
        {reinterpret_cast<ElementC*>(d_C), N},
        {reinterpret_cast<ElementC*>(d_C), N},
        {cutlass::half_t(1.0f), cutlass::half_t(0.0f)}
    );

    GemmNN op;
    auto status = op.initialize(args, nullptr, nullptr);
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS initialize failed\n");
        return 1;
    }

    // Extract params from CUTLASS op object
    using Params = typename GemmNN::GemmKernel::Params;
    uint8_t cutlass_params[512] = {};
    memcpy(cutlass_params, &op, sizeof(Params));

    // Method 2: Our bridge
    uint8_t bridge_params[512] = {};
    int grid[3], block[3], smem;
    int psz = cutlass_params_nn_64x64_64_s6(
        bridge_params, 512, grid, block, &smem,
        N, M, K,
        (uint64_t)d_B, N, (uint64_t)d_A, K, (uint64_t)d_C, N,
        0x3C00, 0x0000);

    printf("\nsizeof(Params) = %zu, bridge psz = %d\n", sizeof(Params), psz);

    // Compare byte-by-byte
    int diffs = 0;
    for (int i = 0; i < (int)sizeof(Params); i++) {
        if (cutlass_params[i] != bridge_params[i]) {
            if (diffs < 20)
                printf("  DIFF at byte %3d (0x%03x): cutlass=0x%02x bridge=0x%02x\n",
                       i, i, cutlass_params[i], bridge_params[i]);
            diffs++;
        }
    }

    if (diffs == 0) printf("\nParams are IDENTICAL (%zu bytes)\n", sizeof(Params));
    else printf("\n%d bytes differ out of %zu\n", diffs, sizeof(Params));

    // Dump both as hex for visual comparison
    printf("\n=== CUTLASS internal params ===\n");
    for (int row = 0; row < (int)sizeof(Params); row += 32) {
        printf("  [%03x]: ", row);
        for (int i = row; i < row + 32 && i < (int)sizeof(Params); i += 4)
            printf("%08x ", *(uint32_t*)(cutlass_params + i));
        printf("\n");
    }
    printf("\n=== Bridge params ===\n");
    for (int row = 0; row < psz; row += 32) {
        printf("  [%03x]: ", row);
        for (int i = row; i < row + 32 && i < psz; i += 4)
            printf("%08x ", *(uint32_t*)(bridge_params + i));
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return diffs > 0 ? 1 : 0;
}
