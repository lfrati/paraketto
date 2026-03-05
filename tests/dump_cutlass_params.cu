// dump_cutlass_params.cu — Dump CUTLASS Params bytes for comparison
// Compile: nvcc -std=c++17 -O3 -arch=sm_80 --expt-relaxed-constexpr -Ithird_party/cutlass/include -Isrc -I/usr/local/cuda-13.1/include tests/dump_cutlass_params.cu -o dump_cutlass_params -lcudart
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

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

int main() {
    using Params = typename GemmNN::GemmKernel::Params;
    printf("sizeof(GemmNN) = %zu\n", sizeof(GemmNN));
    printf("sizeof(Params) = %zu\n", sizeof(Params));
    printf("kThreadCount = %d\n", GemmNN::GemmKernel::kThreadCount);
    printf("SharedStorage = %zu\n", sizeof(typename GemmNN::GemmKernel::SharedStorage));

    // Use FAKE GPU addresses that match the cudaless test pattern
    uint64_t fakeA = 0x100c27d000ULL;
    uint64_t fakeB = 0x100c27b000ULL;
    uint64_t fakeC = 0x100c27f000ULL;

    using Element = cutlass::half_t;
    cutlass::half_t alpha, beta;
    uint16_t a16 = 0x3C00, b16 = 0x0000;
    memcpy(&alpha, &a16, 2);
    memcpy(&beta, &b16, 2);

    // Construct params the same way as cutlass_params.cu
    // Row-major API: Y[64,64] = X[64,64] @ W[64,64] (NN)
    // Col-major conversion: M_cutlass=64, N_cutlass=64, K=64
    // A_cutlass=W, B_cutlass=X, C_cutlass=Y
    int M = 64, N = 64, K = 64;
    GemmNN::Arguments args(
        {M, N, K},
        {reinterpret_cast<const Element*>(fakeA), M},  // A: ldA=M for col-major
        {reinterpret_cast<const Element*>(fakeB), K},   // B: ldB=K for col-major NN
        {reinterpret_cast<Element*>(fakeC), M},          // C: ldC=M for col-major
        {reinterpret_cast<Element*>(fakeC), M},          // D: same as C
        {alpha, beta}
    );

    GemmNN op;
    auto status = op.initialize(args, nullptr, nullptr);
    if (status != cutlass::Status::kSuccess) {
        printf("initialize failed: %d\n", (int)status);
        return 1;
    }

    // Dump params
    const uint8_t* bytes = (const uint8_t*)&op;
    int psz = sizeof(Params);
    printf("\nParams (%d bytes):\n", psz);
    for (int row = 0; row < psz; row += 32) {
        printf("  [%03x]: ", row);
        for (int i = row; i < row + 32 && i < psz; i += 4)
            printf("%08x ", *(const uint32_t*)(bytes + i));
        printf("\n");
    }

    // Also construct with the ACTUAL args the cudaless test uses
    // In the cudaless test, gemm_nn calls:
    //   launch_gemm(NN, n, m, k, W, n, X, k, Y, n, ...)
    // For M_test=64, N_test=64, K_test=64:
    //   launch_gemm M=64, N=64, K=64, A=W(ldA=64), B=X(ldB=64), C=Y(ldC=64)
    printf("\n--- With ldA=64, ldB=64, ldC=64 (actual cudaless args) ---\n");
    GemmNN::Arguments args2(
        {64, 64, 64},
        {reinterpret_cast<const Element*>(fakeA), 64},
        {reinterpret_cast<const Element*>(fakeB), 64},
        {reinterpret_cast<Element*>(fakeC), 64},
        {reinterpret_cast<Element*>(fakeC), 64},
        {alpha, beta}
    );
    GemmNN op2;
    status = op2.initialize(args2, nullptr, nullptr);
    if (status != cutlass::Status::kSuccess) {
        printf("initialize2 failed: %d\n", (int)status);
        return 1;
    }
    bytes = (const uint8_t*)&op2;
    printf("Params (%d bytes):\n", psz);
    for (int row = 0; row < psz; row += 32) {
        printf("  [%03x]: ", row);
        for (int i = row; i < row + 32 && i < psz; i += 4)
            printf("%08x ", *(const uint32_t*)(bytes + i));
        printf("\n");
    }

    return 0;
}
