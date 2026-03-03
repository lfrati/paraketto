// sniff_cutlass2.cu — Dump CUTLASS Params struct layout by inspecting sizeof and using
// a device-side kernel to write the Params bytes to a buffer we can read back.
//
// Approach: instead of scanning host memory for QMD, we use CUTLASS's own API to
// construct the Params, then memcpy the raw bytes and analyze them.

#include "cutlass_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Redefine the CUTLASS types from cutlass_gemm.cu
template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using BatchSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

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

using GemmTN = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 6, 8, 8>;

using BatchNN = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, BatchSwizzle, 6, 8, 8>;

using BatchTN = cutlass::gemm::device::GemmBatched<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, BatchSwizzle, 6, 8, 8>;

template<typename GemmOp>
static void dump_gemm_params(const char* name, int M, int N, int K,
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
    auto status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "  %s: initialize failed: %d\n", name, (int)status);
        return;
    }

    // Access the internal params_ by treating the object as raw bytes
    // device::Gemm stores params_ as the first member
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&op);
    size_t op_size = sizeof(GemmOp);

    printf("\n=== %s ===\n", name);
    printf("  sizeof(GemmOp)=%zu  sizeof(Params)=%zu\n",
           op_size, sizeof(typename GemmOp::GemmKernel::Params));

    using Params = typename GemmOp::GemmKernel::Params;
    size_t params_size = sizeof(Params);

    // Dump raw bytes searching for known values
    uint64_t ptr_a = (uint64_t)A, ptr_b = (uint64_t)B, ptr_c = (uint64_t)C;
    printf("  Known: A=0x%lx B=0x%lx C=0x%lx M=%d N=%d K=%d ldA=%d ldB=%d ldC=%d\n",
           (unsigned long)ptr_a, (unsigned long)ptr_b, (unsigned long)ptr_c,
           M, N, K, ldA, ldB, ldC);

    // The Params are the first bytes in the GemmOp struct
    printf("  Params raw dump (%zu bytes):\n", params_size);
    for (size_t i = 0; i < params_size; i += 8) {
        uint64_t v64 = 0;
        uint32_t v32_lo = 0, v32_hi = 0;
        memcpy(&v64, raw + i, (i + 8 <= params_size) ? 8 : params_size - i);
        memcpy(&v32_lo, raw + i, 4);
        if (i + 4 < params_size) memcpy(&v32_hi, raw + i + 4, 4);

        printf("    [%3zu] 0x%08x 0x%08x", i, v32_lo, v32_hi);

        if (v64 == ptr_a) printf("  ← ptr_A");
        else if (v64 == ptr_b) printf("  ← ptr_B");
        else if (v64 == ptr_c) printf("  ← ptr_C");

        if (v32_lo == (uint32_t)M) printf("  lo=M");
        if (v32_lo == (uint32_t)N) printf("  lo=N");
        if (v32_lo == (uint32_t)K) printf("  lo=K");
        if (v32_lo == (uint32_t)ldA) printf("  lo=ldA");
        if (v32_lo == (uint32_t)ldB) printf("  lo=ldB");
        if (v32_lo == (uint32_t)ldC) printf("  lo=ldC");
        if (v32_hi == (uint32_t)M) printf("  hi=M");
        if (v32_hi == (uint32_t)N) printf("  hi=N");
        if (v32_hi == (uint32_t)K) printf("  hi=K");
        if (v32_hi == (uint32_t)ldA) printf("  hi=ldA");
        if (v32_hi == (uint32_t)ldB) printf("  hi=ldB");
        if (v32_hi == (uint32_t)ldC) printf("  hi=ldC");

        // Check fp16 alpha/beta
        uint16_t h0, h1;
        memcpy(&h0, raw + i, 2);
        memcpy(&h1, raw + i + 2, 2);
        if (h0 == 0x3C00) printf("  h0=1.0h");
        if (h1 == 0x3C00) printf("  h1=1.0h");

        printf("\n");
    }
}

template<typename BatchedGemmOp>
static void dump_batched_params(const char* name, int M, int N, int K,
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
    auto status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "  %s: initialize failed: %d\n", name, (int)status);
        return;
    }

    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&op);
    using Params = typename BatchedGemmOp::GemmKernel::Params;
    size_t params_size = sizeof(Params);

    uint64_t ptr_a = (uint64_t)A, ptr_b = (uint64_t)B, ptr_c = (uint64_t)C;

    printf("\n=== %s ===\n", name);
    printf("  sizeof(GemmOp)=%zu  sizeof(Params)=%zu\n", sizeof(BatchedGemmOp), params_size);
    printf("  Known: A=0x%lx B=0x%lx C=0x%lx M=%d N=%d K=%d batch=%d\n",
           (unsigned long)ptr_a, (unsigned long)ptr_b, (unsigned long)ptr_c, M, N, K, batch);
    printf("  ldA=%d ldB=%d ldC=%d strA=%lld strB=%lld strC=%lld\n",
           ldA, ldB, ldC, strideA, strideB, strideC);

    printf("  Params raw dump (%zu bytes):\n", params_size);
    for (size_t i = 0; i < params_size; i += 8) {
        uint64_t v64 = 0;
        uint32_t v32_lo = 0, v32_hi = 0;
        memcpy(&v64, raw + i, (i + 8 <= params_size) ? 8 : params_size - i);
        memcpy(&v32_lo, raw + i, 4);
        if (i + 4 < params_size) memcpy(&v32_hi, raw + i + 4, 4);

        printf("    [%3zu] 0x%08x 0x%08x", i, v32_lo, v32_hi);

        if (v64 == ptr_a) printf("  ← ptr_A");
        else if (v64 == ptr_b) printf("  ← ptr_B");
        else if (v64 == ptr_c) printf("  ← ptr_C");
        if ((long long)v64 == strideA) printf("  ← strideA");
        if ((long long)v64 == strideB) printf("  ← strideB");
        if ((long long)v64 == strideC) printf("  ← strideC");

        if (v32_lo == (uint32_t)M) printf("  lo=M");
        if (v32_lo == (uint32_t)N) printf("  lo=N");
        if (v32_lo == (uint32_t)K) printf("  lo=K");
        if (v32_lo == (uint32_t)batch) printf("  lo=batch");
        if (v32_lo == (uint32_t)ldA) printf("  lo=ldA");
        if (v32_lo == (uint32_t)ldB) printf("  lo=ldB");
        if (v32_lo == (uint32_t)ldC) printf("  lo=ldC");
        if (v32_hi == (uint32_t)M) printf("  hi=M");
        if (v32_hi == (uint32_t)N) printf("  hi=N");
        if (v32_hi == (uint32_t)K) printf("  hi=K");
        if (v32_hi == (uint32_t)batch) printf("  hi=batch");

        printf("\n");
    }
}

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Use distinctive sizes
    int M = 63, N = 128, K = 256;
    half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * 2));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * 2));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * 2));

    // NN: col_m=N=128, col_n=M=63, col_k=K=256
    // CUTLASS A=d_B(ldA=N=128), B=d_A(ldB=K=256), C=d_C(ldC=N=128)
    dump_gemm_params<GemmNN>("NN 64x64x64 s6",
        N, M, K, (const half*)d_B, N, (const half*)d_A, K, d_C, N, alpha, beta, stream);

    dump_gemm_params<GemmTN>("TN 64x64x64 s6",
        N, M, K, (const half*)d_B, K, (const half*)d_A, K, d_C, N, alpha, beta, stream);

    // Batched
    int batch = 4;
    long long sA = M * K, sB = K * N, sC = M * N;
    half *d_bA, *d_bB, *d_bC;
    CUDA_CHECK(cudaMalloc(&d_bA, batch * sA * 2));
    CUDA_CHECK(cudaMalloc(&d_bB, batch * sB * 2));
    CUDA_CHECK(cudaMalloc(&d_bC, batch * sC * 2));

    dump_batched_params<BatchNN>("Batched NN 64x64x64 s6",
        N, M, K,
        (const half*)d_bB, N, sB, (const half*)d_bA, K, sA, d_bC, N, sC,
        batch, alpha, beta, stream);

    dump_batched_params<BatchTN>("Batched TN 64x64x64 s6",
        N, M, K,
        (const half*)d_bB, K, sB, (const half*)d_bA, K, sA, d_bC, N, sC,
        batch, alpha, beta, stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    printf("\nDone.\n");
    return 0;
}
