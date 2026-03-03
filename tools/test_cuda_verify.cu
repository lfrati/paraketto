// test_cuda_verify.cu — verify add_relu_kernel works via CUDA runtime
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>

__global__ void add_relu_kernel(const half* a, const half* b, half* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float va = __half2float(a[i]);
    float vb = __half2float(b[i]);
    y[i] = __float2half(fmaxf(va + vb, 0.0f));
}

int main() {
    const int N = 4096;
    half *d_a, *d_b, *d_y;
    half *h_a = new half[N], *h_b = new half[N], *h_y = new half[N];

    srand(42);
    for (int i = 0; i < N; i++) {
        float va = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        float vb = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_a[i] = __float2half(va);
        h_b[i] = __float2half(vb);
        h_y[i] = __float2half(-1.7334f); // 0xBEEF equivalent
    }

    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_a, h_a, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(half), cudaMemcpyHostToDevice);

    add_relu_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float fa = __half2float(h_a[i]);
        float fb = __half2float(h_b[i]);
        float expected = fmaxf(fa + fb, 0.0f);
        float got = __half2float(h_y[i]);
        if (fabsf(got - expected) > 0.01f) {
            if (mismatches < 5)
                fprintf(stderr, "MISMATCH i=%d: expected=%.4f got=%.4f\n", i, expected, got);
            mismatches++;
        }
    }

    if (mismatches == 0)
        fprintf(stderr, "[PASS] add_relu via CUDA: N=%d, 0 mismatches\n", N);
    else
        fprintf(stderr, "[FAIL] add_relu via CUDA: N=%d, %d mismatches\n", N, mismatches);

    // Now dump what CUDA puts in cbuf0 at key offsets
    // This requires inspecting the kernel launch params - can't do directly
    // But at least we know the kernel code works

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
    delete[] h_a; delete[] h_b; delete[] h_y;
    return mismatches ? 1 : 0;
}
