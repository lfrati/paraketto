// test_cutlass_cuda_vs_cudaless.cu — Verify CUTLASS works via CUDA runtime
// This confirms the cubin code itself is correct; any failure in cudaless is launch-side.
//
// Build: nvcc -std=c++17 -O3 -arch=sm_120 --expt-relaxed-constexpr
//   -Ithird_party/cutlass/include -Isrc tests/test_cutlass_cuda_vs_cudaless.cu -o test_cutlass_cuda -lcudart

#include "cutlass_gemm.h"  // The CUDA-based CUTLASS wrappers
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>

static void fill_random(half* data, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        float val = ((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f;
        data[i] = __float2half(val);
    }
}

static void cpu_gemm_nn(const half* A, const half* B, half* C, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++)
                sum += __half2float(A[i*k+p]) * __half2float(B[p*n+j]);
            C[i*n+j] = __float2half(sum);
        }
}

int main() {
    printf("=== CUTLASS CUDA Runtime Test ===\n\n");

    const int M = 64, N = 64, K = 64;
    size_t sA = M*K*2, sB = K*N*2, sC = M*N*2;

    // Allocate host
    half *h_A = (half*)malloc(sA), *h_B = (half*)malloc(sB);
    half *h_C = (half*)malloc(sC), *cpu_C = (half*)malloc(sC);
    fill_random(h_A, M*K, 42);
    fill_random(h_B, K*N, 123);
    memset(h_C, 0, sC);

    // CPU reference
    cpu_gemm_nn(h_A, h_B, cpu_C, M, N, K);

    // Allocate GPU
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sA);
    cudaMalloc(&d_B, sB);
    cudaMalloc(&d_C, sC);
    cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sC);

    // Launch CUTLASS NN via CUDA runtime
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cutlass_gemm_init(stream);
    cutlass_gemm_nn(stream, d_A, M, K, d_B, N, d_C);
    cudaStreamSynchronize(stream);
    cutlass_gemm_free();
    cudaStreamDestroy(stream);

    // Read back
    cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost);

    // Compare
    int errors = 0;
    float max_rel = 0;
    for (int i = 0; i < M*N; i++) {
        float g = __half2float(h_C[i]);
        float c = __half2float(cpu_C[i]);
        float diff = fabsf(g - c);
        float rel = (fabsf(c) > 1e-6f) ? diff / fabsf(c) : diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > 0.01f && rel > 0.02f) {
            if (errors < 5)
                printf("  MISMATCH [%d]: gpu=%.6f cpu=%.6f diff=%.6f rel=%.4f\n", i, g, c, diff, rel);
            errors++;
        }
    }
    if (errors == 0)
        printf("gemm_nn 64x64x64 via CUDA: PASS (max_rel=%.4f)\n", max_rel);
    else
        printf("gemm_nn 64x64x64 via CUDA: FAIL (%d/%d errors, max_rel=%.4f)\n", errors, M*N, max_rel);

    free(h_A); free(h_B); free(h_C); free(cpu_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return errors > 0 ? 1 : 0;
}
