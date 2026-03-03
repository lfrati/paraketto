// Test: can we construct CUTLASS Params with g++ (no nvcc, no libcudart)?
// Just include CUTLASS headers with CUDA include path for types.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

int main() {
    printf("sizeof(half) = %zu\n", sizeof(half));
    printf("CUTLASS OK\n");
    return 0;
}
