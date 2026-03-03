// sniff_qmd2.cu — Launch kernel via CUDA, dump ALL 96 QMD DWORDs
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

struct MappedRegion { uintptr_t start, end; bool readable; };

static std::vector<MappedRegion> get_mapped_regions() {
    std::vector<MappedRegion> regions;
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) return regions;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        uintptr_t start, end; char perms[8];
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

    add_relu_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    auto regions = get_mapped_regions();
    for (auto& r : regions) {
        if (!r.readable) continue;
        size_t size = r.end - r.start;
        if (size < 384 || size > 256*1024*1024) continue;
        const uint32_t* base = (const uint32_t*)r.start;
        size_t n_dwords = size / 4;
        for (size_t i = 0; i + 96 <= n_dwords; i++) {
            const uint32_t* dw = base + i;
            if (((dw[14] >> 20) & 0xF) != 5) continue;
            if ((dw[14] & 0xFF) != 0xA4) continue;
            if (((dw[4] >> 23) & 0x7) != 2) continue;
            if (dw[39] != (N+255)/256 || (dw[34] & 0xFFFF) != 256) continue;

            // Print ALL 96 DWORDs
            fprintf(stderr, "CUDA QMD at %p:\n", (void*)dw);
            for (int j = 0; j < 96; j += 4)
                fprintf(stderr, "  DW[%2d-%2d]: %08x %08x %08x %08x\n",
                        j, j+3, dw[j], dw[j+1], dw[j+2], dw[j+3]);

            // Decode specific fields
            uint64_t prog = (((uint64_t)(dw[33] & 0x1FFFFF) << 32) | dw[32]) << 4;
            uint32_t prefetch_size = (dw[33] >> 21) & 0x1FF;
            uint32_t prefetch_type = (dw[33] >> 30) & 0x3;
            uint32_t local_lo = dw[37] & 0xFFFF;
            uint32_t local_hi = (dw[37] >> 16) & 0xFFFF;

            fprintf(stderr, "\nDecoded fields:\n");
            fprintf(stderr, "  prog_addr = 0x%lx\n", (unsigned long)prog);
            fprintf(stderr, "  DW[33] raw = 0x%08x\n", dw[33]);
            fprintf(stderr, "  prefetch_size_s8 = %u (= %u bytes)\n", prefetch_size, prefetch_size << 8);
            fprintf(stderr, "  prefetch_type = %u (0=none, 1=CTA)\n", prefetch_type);
            fprintf(stderr, "  DW[37] raw = 0x%08x\n", dw[37]);
            fprintf(stderr, "  local_mem_low_s4 = 0x%x (= %u bytes)\n", local_lo, local_lo << 4);
            fprintf(stderr, "  local_mem_hi_s4  = 0x%x (= %u bytes)\n", local_hi, local_hi << 4);

            goto done;
        }
    }
    fprintf(stderr, "No QMD found!\n");
done:
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
    delete[] h_a;
    return 0;
}
