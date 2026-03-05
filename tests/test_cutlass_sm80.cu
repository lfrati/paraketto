// Verify CUTLASS works via normal CUDA path (sm_80 .o, cudaLaunchKernel)
#include "cutlass_gemm.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>

static float h2f(half h) { return __half2float(h); }
static half f2h(float f) { return __float2half(f); }

int main() {
    const int M = 64, N = 64, K = 64;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cutlass_gemm_init(stream);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * 2);
    cudaMalloc(&d_B, K * N * 2);
    cudaMalloc(&d_C, M * N * 2);

    // Same random data as cudaless test
    half h_A[M * K], h_B[K * N];
    unsigned seed = 42;
    for (int i = 0; i < M * K; i++) {
        seed = seed * 1103515245 + 12345;
        h_A[i] = f2h(((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f);
    }
    seed = 123;
    for (int i = 0; i < K * N; i++) {
        seed = seed * 1103515245 + 12345;
        h_B[i] = f2h(((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f);
    }
    cudaMemcpy(d_A, h_A, M * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * 2);

    // Launch via normal CUDA path
    cutlass_gemm_nn(stream, d_A, M, K, d_B, N, d_C);
    cudaStreamSynchronize(stream);

    half h_C[M * N];
    cudaMemcpy(h_C, d_C, M * N * 2, cudaMemcpyDeviceToHost);

    // CPU reference
    float cpu_C[M * N] = {};
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int p = 0; p < K; p++)
                cpu_C[i * N + j] += h2f(h_A[i * K + p]) * h2f(h_B[p * N + j]);

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float g = h2f(h_C[i]);
        float c = cpu_C[i];
        float diff = fabsf(g - c);
        float rel = (fabsf(c) > 1e-6f) ? diff / fabsf(c) : diff;
        if (diff > 0.5f && rel > 0.05f) {
            if (errors < 5) printf("  [%d] gpu=%.4f cpu=%.4f\n", i, g, c);
            errors++;
        }
    }

    printf("SM80 CUDA path: %d/%d errors\n", errors, M * N);
    printf("First 8 GPU: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h2f(h_C[i]));
    printf("\nFirst 8 CPU: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", cpu_C[i]);
    printf("\n");

    cutlass_gemm_free();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaStreamDestroy(stream);
    return errors > 0 ? 1 : 0;
}
