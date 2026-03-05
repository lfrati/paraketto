// Dump the SM120 generic address descriptor from cbuf0[0x358]
// The kernel reads it from constant memory and stores it via a known-working path
#include <cstdio>
#include <cstdint>

// The descriptor at c[0x0][0x358] is needed for desc[URx] memory operations on SM120.
// We'll use threadIdx and blockIdx (which don't need descriptors) to output the value.
// Strategy: allocate shared memory, write descriptor there, then use atomicAdd to output.

__global__ void dump_desc_kernel(uint32_t* out) {
    // SM120 stores a 64-bit generic addr descriptor at c[0x0][0x358]
    // We need to read it and output it. But we can't use printf (needs global mem desc).
    // Instead, we'll use atomicAdd on the out pointer, which uses the desc.
    // Catch-22: we need the desc to write the desc!
    //
    // BUT: the CUDA runtime sets up the desc correctly before launching.
    // So global stores WILL work in this kernel. We just read the desc and store it.

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Read cbuf0 words around 0x358
        // We can't use ld.const directly. But the compiler loads it automatically
        // for any global memory access. Let's just read the desc by using
        // a pointer to the constant buffer.

        // Actually, just store the gridDim values which come from known cbuf0 offsets
        out[0] = gridDim.x;  // from c[0x0][0x35c] typically
        out[1] = gridDim.y;
        out[2] = gridDim.z;
        out[3] = blockDim.x;
        out[4] = blockDim.y;
        out[5] = blockDim.z;

        // Store a known pattern so we know the kernel ran
        out[6] = 0xDEADBEEF;
    }
}

// Use PTX to directly read the constant buffer
__global__ void dump_desc_ptx(uint64_t* out) {
    if (threadIdx.x != 0) return;

    // Use mov to get special register values that come from cbuf0
    uint32_t nctaid_x, nctaid_y, nctaid_z;
    uint32_t ntid_x, ntid_y, ntid_z;
    asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(nctaid_x));
    asm volatile("mov.u32 %0, %%nctaid.y;" : "=r"(nctaid_y));
    asm volatile("mov.u32 %0, %%nctaid.z;" : "=r"(nctaid_z));
    asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(ntid_x));
    asm volatile("mov.u32 %0, %%ntid.y;" : "=r"(ntid_y));
    asm volatile("mov.u32 %0, %%ntid.z;" : "=r"(ntid_z));

    out[0] = nctaid_x | ((uint64_t)nctaid_y << 32);
    out[1] = nctaid_z | ((uint64_t)ntid_x << 32);
    out[2] = ntid_y | ((uint64_t)ntid_z << 32);
    out[3] = 0xCAFEBABE;
}

int main() {
    uint64_t *d_out;
    uint64_t h_out[16] = {};
    cudaMalloc(&d_out, 128);
    cudaMemset(d_out, 0, 128);

    dump_desc_kernel<<<2, 32>>>((uint32_t*)d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128, cudaMemcpyDeviceToHost);

    uint32_t* w = (uint32_t*)h_out;
    printf("gridDim:  (%u, %u, %u)\n", w[0], w[1], w[2]);
    printf("blockDim: (%u, %u, %u)\n", w[3], w[4], w[5]);
    printf("marker:   0x%08x\n", w[6]);

    // Now try the PTX version
    cudaMemset(d_out, 0, 128);
    dump_desc_ptx<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128, cudaMemcpyDeviceToHost);
    printf("\nPTX version:\n");
    printf("nctaid: (%u, %u)\n", (uint32_t)h_out[0], (uint32_t)(h_out[0]>>32));
    printf("nctaid.z=%u ntid.x=%u\n", (uint32_t)h_out[1], (uint32_t)(h_out[1]>>32));
    printf("marker: 0x%lx\n", h_out[3]);

    cudaFree(d_out);
    return 0;
}
