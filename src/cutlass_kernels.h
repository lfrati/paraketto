// cutlass_kernels.h — Fused GEMM+activation kernels using CUTLASS
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// GEMM + SiLU: Y = SiLU(X @ W) where W is [K, N] (ONNX MatMul convention)
void gemm_silu_nn_fp16(const half* X, int M, int K,
                        const half* W, int N, half* Y,
                        cudaStream_t stream);

// GEMM + SiLU: Y = SiLU(X @ W^T) where W is [N, K] (Conv/PyTorch convention)
void gemm_silu_nt_fp16(const half* X, int M, int K,
                        const half* W, int N, half* Y,
                        cudaStream_t stream);
