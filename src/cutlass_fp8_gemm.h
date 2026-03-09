// cutlass_fp8_gemm.h — CUTLASS 3.x SM120 FP8 E4M3 GEMM wrappers
//
// Replaces cublasLt FP8 GEMMs with custom CUTLASS kernels using trivial
// blockwise scaling (all scale blocks = 1.0f, alpha = act_scale * wt_scale).
//
// SM120 blockwise MMA reads B as [N,K] row-major (K contiguous).
// The caller must pre-transpose NN weights from [K,N] to [N,K].
// NT weights are naturally [N,K] and can be passed directly.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Initialize CUTLASS FP8 module: allocate scale buffers (filled with 1.0f)
// and workspace for the largest GEMM shape.
void cutlass_fp8_init(int max_M, int max_N, int max_K, cudaStream_t stream);

// Free all CUTLASS FP8 resources.
void cutlass_fp8_free();

// Y[M,N] = alpha * A_fp8[M,K] @ B_fp8[N,K]^T
// A: [M,K] row-major FP8, B: [N,K] row-major FP8, Y: [M,N] row-major FP16
// Returns 0 on success, negative on error.
int cutlass_fp8_nn(cudaStream_t stream, int M, int N, int K,
                   const uint8_t* A, const uint8_t* B, half* C, float alpha);
