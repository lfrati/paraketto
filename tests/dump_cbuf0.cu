// dump_cbuf0.cu — Kernel that dumps cbuf0 driver parameters via inline PTX
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

// Kernel that reads specific cbuf0 offsets and stores to output
__global__ void dump_cbuf0_kernel(uint32_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Read various cbuf0 offsets using inline PTX
    // These are the "driver parameter" area that CUDA fills before launch
    uint32_t val;

    // Dump DWORDs from 0x340 to 0x380 (offsets 832-895)
    // This covers the descriptor area and blockDim
    #define DUMP(offset, idx) \
        asm volatile("ld.const.u32 %0, [" #offset "];" : "=r"(val)); \
        out[idx] = val;

    DUMP(0x340, 0)   DUMP(0x344, 1)   DUMP(0x348, 2)   DUMP(0x34c, 3)
    DUMP(0x350, 4)   DUMP(0x354, 5)   DUMP(0x358, 6)   DUMP(0x35c, 7)
    DUMP(0x360, 8)   DUMP(0x364, 9)   DUMP(0x368, 10)  DUMP(0x36c, 11)
    DUMP(0x370, 12)  DUMP(0x374, 13)  DUMP(0x378, 14)  DUMP(0x37c, 15)

    // Also dump the first few DWORDs and the known driver params
    DUMP(0x000, 16)  DUMP(0x004, 17)  DUMP(0x008, 18)  DUMP(0x00c, 19)

    // DWORDs 188-191 (shared/local windows at 0x2F0-0x2FF)
    DUMP(0x2F0, 20)  DUMP(0x2F4, 21)  DUMP(0x2F8, 22)  DUMP(0x2FC, 23)

    // DWORD 223 (at 0x37C) already covered above as index 15

    // Dump more of the descriptor area
    DUMP(0x300, 24)  DUMP(0x304, 25)  DUMP(0x308, 26)  DUMP(0x30c, 27)
    DUMP(0x310, 28)  DUMP(0x314, 29)  DUMP(0x318, 30)  DUMP(0x31c, 31)
    DUMP(0x320, 32)  DUMP(0x324, 33)  DUMP(0x328, 34)  DUMP(0x32c, 35)
    DUMP(0x330, 36)  DUMP(0x334, 37)  DUMP(0x338, 38)  DUMP(0x33c, 39)
}

// Also dump cbuf0 for add_relu specifically
__global__ void add_relu_kernel(const half* a, const half* b, half* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float va = __half2float(a[i]);
    float vb = __half2float(b[i]);
    y[i] = __float2half(fmaxf(va + vb, 0.0f));
}

// Kernel that reads cbuf0 values right before add_relu would
__global__ void probe_add_relu(uint32_t* out, const half* a, const half* b, half* y, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t val;
    // Read descriptor at 0x358 (what add_relu's LDCU.64 UR4 reads)
    DUMP(0x358, 0)   DUMP(0x35c, 1)
    // Read blockDim at 0x360
    DUMP(0x360, 2)   DUMP(0x364, 3)
    // Stack pointer at 0x37c
    DUMP(0x37c, 4)
    // Kernel args start at 0x380 for add_relu
    DUMP(0x380, 5)   DUMP(0x384, 6)   // a_ptr
    DUMP(0x388, 7)   DUMP(0x38c, 8)   // b_ptr
    DUMP(0x390, 9)   DUMP(0x394, 10)  // y_ptr
    DUMP(0x398, 11)                    // n
}

int main() {
    uint32_t *d_out, h_out[64] = {};
    cudaMalloc(&d_out, 256);
    cudaMemset(d_out, 0, 256);

    // Launch dump kernel
    dump_cbuf0_kernel<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 160, cudaMemcpyDeviceToHost);

    fprintf(stderr, "=== cbuf0 dump from dump_cbuf0_kernel ===\n");
    for (int i = 0; i < 16; i++) {
        fprintf(stderr, "  c[0x0][0x%03x] = 0x%08x", 0x340 + i*4, h_out[i]);
        if (i == 6) fprintf(stderr, "  <-- descriptor low (0x358)");
        if (i == 7) fprintf(stderr, "  <-- descriptor high (0x35c)");
        if (i == 8) fprintf(stderr, "  <-- blockDim.x (0x360)");
        if (i == 15) fprintf(stderr, " <-- 0x37c (stack?)");
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\nFirst 4 DWORDs:\n");
    for (int i = 16; i < 20; i++)
        fprintf(stderr, "  c[0x0][0x%03x] = 0x%08x\n", (i-16)*4, h_out[i]);

    fprintf(stderr, "\nShared/Local windows (DW 188-191):\n");
    for (int i = 20; i < 24; i++)
        fprintf(stderr, "  c[0x0][0x%03x] = 0x%08x\n", 0x2F0 + (i-20)*4, h_out[i]);

    fprintf(stderr, "\ncbuf0 0x300-0x33C:\n");
    for (int i = 24; i < 40; i++)
        fprintf(stderr, "  c[0x0][0x%03x] = 0x%08x\n", 0x300 + (i-24)*4, h_out[i]);

    // Now launch probe_add_relu with same args as test
    half *d_a, *d_b, *d_y;
    const int N = 256;
    cudaMalloc(&d_a, N*2);
    cudaMalloc(&d_b, N*2);
    cudaMalloc(&d_y, N*2);
    cudaMemset(d_out, 0, 256);

    probe_add_relu<<<1, 256>>>(d_out, d_a, d_b, d_y, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 48, cudaMemcpyDeviceToHost);

    fprintf(stderr, "\n=== cbuf0 from probe_add_relu (same args as add_relu) ===\n");
    fprintf(stderr, "  descriptor: 0x%08x_%08x\n", h_out[1], h_out[0]);
    fprintf(stderr, "  blockDim:   0x%08x_%08x\n", h_out[3], h_out[2]);
    fprintf(stderr, "  stack:      0x%08x\n", h_out[4]);
    fprintf(stderr, "  a_ptr:      0x%08x_%08x\n", h_out[6], h_out[5]);
    fprintf(stderr, "  b_ptr:      0x%08x_%08x\n", h_out[8], h_out[7]);
    fprintf(stderr, "  y_ptr:      0x%08x_%08x\n", h_out[10], h_out[9]);
    fprintf(stderr, "  n:          %d\n", h_out[11]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y); cudaFree(d_out);
    return 0;
}
