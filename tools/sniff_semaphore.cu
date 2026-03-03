// sniff_semaphore.cu — Launch kernel via CUDA, scan memory for QMD release semaphore
// Focus: find the exact semaphore bytes CUDA writes and where
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>

__global__ void add_relu_kernel(const half* a, const half* b, half* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float va = __half2float(a[i]);
    float vb = __half2float(b[i]);
    y[i] = __float2half(fmaxf(va + vb, 0.0f));
}

struct MappedRegion {
    uintptr_t start, end;
    bool readable;
};

static std::vector<MappedRegion> get_mapped_regions() {
    std::vector<MappedRegion> regions;
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) return regions;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        uintptr_t start, end;
        char perms[8];
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) == 3)
            regions.push_back({start, end, perms[0] == 'r'});
    }
    fclose(f);
    return regions;
}

int main() {
    const int N = 4096;
    half *d_a, *d_b, *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));

    half* h_a = new half[N];
    srand(42);
    for (int i = 0; i < N; i++)
        h_a[i] = __float2half(((float)rand() / RAND_MAX) * 4.0f - 2.0f);
    cudaMemcpy(d_a, h_a, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, N * sizeof(half), cudaMemcpyHostToDevice);

    fprintf(stderr, "GPU pointers: a=%p b=%p y=%p  n=%d\n", d_a, d_b, d_y, N);

    // Scan BEFORE launch to get baseline
    auto regions_before = get_mapped_regions();
    fprintf(stderr, "Regions before launch: %zu\n", regions_before.size());

    add_relu_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    fprintf(stderr, "\n=== Post-launch: Scanning for QMD v5 to extract semaphore info ===\n");
    auto regions = get_mapped_regions();

    for (auto& r : regions) {
        if (!r.readable) continue;
        size_t size = r.end - r.start;
        if (size < 384) continue;
        if (size > 256 * 1024 * 1024) continue;

        const uint32_t* base = (const uint32_t*)r.start;
        size_t n_dwords = size / 4;

        for (size_t i = 0; i + 96 <= n_dwords; i++) {
            const uint32_t* dw = base + i;

            // QMD v5 check
            uint32_t major = (dw[14] >> 20) & 0xF;
            if (major != 5) continue;
            uint32_t sass = dw[14] & 0xFF;
            if (sass != 0xA4) continue;
            uint32_t qtype = (dw[4] >> 23) & 0x7;
            if (qtype != 2) continue;

            uint32_t grid_x = dw[39];
            uint32_t block_x = dw[34] & 0xFFFF;
            if (grid_x != (N+255)/256 || block_x != 256) continue;

            // Extract semaphore info from DW[9] and DW[15-18]
            uint32_t dw9 = dw[9];
            bool release_enable = dw9 & 1;
            uint32_t struct_size = (dw9 >> 1) & 3;
            bool payload64b = (dw9 >> 12) & 1;

            uint64_t sem_addr = ((uint64_t)(dw[16] & 0x1FFFFFF) << 32) | dw[15];
            uint64_t sem_payload = ((uint64_t)dw[18] << 32) | dw[17];

            fprintf(stderr, "\n*** QMD FOUND at %p ***\n", (void*)dw);
            fprintf(stderr, "  DW[9]  = 0x%08x\n", dw9);
            fprintf(stderr, "    RELEASE0_ENABLE = %d\n", release_enable);
            fprintf(stderr, "    RELEASE_STRUCTURE_SIZE = %u (0=16B, 1=4B)\n", struct_size);
            fprintf(stderr, "    RELEASE_PAYLOAD64B = %d\n", payload64b);
            fprintf(stderr, "  DW[15] = 0x%08x (sem_addr_lo)\n", dw[15]);
            fprintf(stderr, "  DW[16] = 0x%08x (sem_addr_hi)\n", dw[16]);
            fprintf(stderr, "  DW[17] = 0x%08x (payload_lo)\n", dw[17]);
            fprintf(stderr, "  DW[18] = 0x%08x (payload_hi)\n", dw[18]);
            fprintf(stderr, "  sem_addr    = 0x%lx\n", (unsigned long)sem_addr);
            fprintf(stderr, "  sem_payload = 0x%lx\n", (unsigned long)sem_payload);

            // Dump ALL of DW[9] neighborhood for context
            fprintf(stderr, "\n  DW[8-11]:  %08x %08x %08x %08x\n",
                    dw[8], dw[9], dw[10], dw[11]);
            fprintf(stderr, "  DW[12-19]: %08x %08x %08x %08x  %08x %08x %08x %08x\n",
                    dw[12], dw[13], dw[14], dw[15],
                    dw[16], dw[17], dw[18], dw[19]);

            // Try to find the semaphore in CPU memory
            fprintf(stderr, "\n  Searching for semaphore at gpu_va=0x%lx...\n",
                    (unsigned long)sem_addr);
            for (auto& r2 : regions) {
                if (!r2.readable) continue;
                size_t sz = r2.end - r2.start;
                if (sz < 16 || sz > 64*1024*1024) continue;
                if (r2.start < 0x10000) continue;

                // Check first 16 bytes for the expected payload
                const uint32_t* p = (const uint32_t*)r2.start;
                size_t ndw = sz / 4;
                for (size_t j = 0; j + 4 <= ndw && j < 65536; j++) {
                    if (p[j] == dw[17] && p[j] != 0) {
                        fprintf(stderr, "    Possible semaphore at %p+%zu: %08x %08x %08x %08x\n",
                                (void*)r2.start, j*4, p[j], p[j+1],
                                j+2 < ndw ? p[j+2] : 0, j+3 < ndw ? p[j+3] : 0);
                    }
                }
            }

            // Now try to find by scanning for any recently changed memory
            // (the semaphore should have been written during cudaDeviceSynchronize)
            goto done;
        }
    }
    fprintf(stderr, "No QMD found!\n");
done:

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
    delete[] h_a;
    return 0;
}
