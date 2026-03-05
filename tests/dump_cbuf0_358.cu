// Dump the value at c[0x0][0x358] that the CUDA runtime fills in cbuf0
// This is the generic address space descriptor used by SM120 for desc[URx] memory access
#include <cstdio>
#include <cstdint>

__global__ void dump_cbuf0(uint64_t* out) {
    uint32_t lo, hi;
    asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(lo));
    asm volatile("mov.u32 %0, %%nctaid.y;" : "=r"(hi));
    out[0] = ((uint64_t)hi << 32) | lo;  // gridDim

    // Read c[0x0][0x358] via inline asm
    // On SM120, this is a 64-bit descriptor loaded by LDCU.64
    uint32_t v358, v35c, v378, v37c;
    asm volatile("ld.const.u32 %0, [0x358];" : "=r"(v358));
    asm volatile("ld.const.u32 %0, [0x35c];" : "=r"(v35c));
    asm volatile("ld.const.u32 %0, [0x378];" : "=r"(v378));
    asm volatile("ld.const.u32 %0, [0x37c];" : "=r"(v37c));
    out[1] = ((uint64_t)v35c << 32) | v358;
    out[2] = ((uint64_t)v37c << 32) | v378;

    // Also read 0x2F0 (shared window) and 0x2F8 (local window)
    uint32_t s0, s1, l0, l1;
    asm volatile("ld.const.u32 %0, [0x2f0];" : "=r"(s0));
    asm volatile("ld.const.u32 %0, [0x2f4];" : "=r"(s1));
    asm volatile("ld.const.u32 %0, [0x2f8];" : "=r"(l0));
    asm volatile("ld.const.u32 %0, [0x2fc];" : "=r"(l1));
    out[3] = ((uint64_t)s1 << 32) | s0;
    out[4] = ((uint64_t)l1 << 32) | l0;

    // Read 0x350-0x370 range for full context
    for (int i = 0; i < 8; i++) {
        uint32_t v;
        asm volatile("ld.const.u32 %0, [%1];" : "=r"(v) : "r"(0x350 + i*4));
        ((uint32_t*)(out + 5))[i] = v;
    }
}

int main() {
    uint64_t *d_out, h_out[16];
    cudaMalloc(&d_out, 128);
    dump_cbuf0<<<1, 1>>>(d_out);
    cudaMemcpy(h_out, d_out, 128, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    printf("gridDim:    0x%016lx\n", h_out[0]);
    printf("c[0][0x358]: 0x%016lx  ← GENERIC ADDR DESC\n", h_out[1]);
    printf("c[0][0x378]: 0x%016lx\n", h_out[2]);
    printf("shared_win: 0x%016lx\n", h_out[3]);
    printf("local_win:  0x%016lx\n", h_out[4]);
    printf("cbuf0[0x350..0x370]:");
    uint32_t* words = (uint32_t*)(h_out + 5);
    for (int i = 0; i < 8; i++)
        printf(" %08x", words[i]);
    printf("\n");
    return 0;
}
