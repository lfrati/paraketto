// gemm.h — Unified GEMM interface for CUTLASS and cuBLAS backends
//
// Row-major convention (matching ONNX/PyTorch):
//   NN: Y[m,n] = X[m,k] @ W[k,n]
//   NT: Y[m,n] = X[m,k] @ W[n,k]^T
//
// Build with -DUSE_CUBLAS to select cuBLAS backend (links -lcublas).
// Default is CUTLASS (header-only, no extra runtime deps).

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Lifecycle
void gemm_init(cudaStream_t stream);
void gemm_free();

// NN: Y[m,n] = X[m,k] @ W[k,n]
void gemm_nn(cudaStream_t stream,
             const half* X, int m, int k,
             const half* W, int n, half* Y);

// NN + bias: Y[m,n] = X[m,k] @ W[k,n] + bias[n]
void gemm_nn_bias(cudaStream_t stream,
                  const half* X, int m, int k,
                  const half* W, int n,
                  const half* bias, half* Y);

// NT: Y[m,n] = X[m,k] @ W[n,k]^T
void gemm_nt(cudaStream_t stream,
             const half* X, int m, int k,
             const half* W, int n, half* Y);

// NT + bias: Y[m,n] = X[m,k] @ W[n,k]^T + bias[n]
void gemm_nt_bias(cudaStream_t stream,
                  const half* X, int m, int k,
                  const half* W, int n,
                  const half* bias, half* Y);

// Batched strided GEMM: C[b,m,n] = A[b,m,k] @ B[b,k,n]
void batched_gemm_nn(cudaStream_t stream,
                     const half* A, const half* B, half* C,
                     int batch, int m, int n, int k,
                     long long strideA, long long strideB, long long strideC);

// Batched strided GEMM: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
void batched_gemm_nt(cudaStream_t stream,
                     const half* A, const half* B, half* C,
                     int batch, int m, int n, int k,
                     long long strideA, long long strideB, long long strideC);

// NT with explicit leading dimensions and strides
void batched_gemm_nt_ex(cudaStream_t stream,
                        const half* A, int ldA, long long strideA,
                        const half* B, int ldB, long long strideB,
                        half* C, int ldC, long long strideC,
                        int batch, int m, int n, int k);
