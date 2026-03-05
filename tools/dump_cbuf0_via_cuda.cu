// Dump cbuf0 by scanning process memory, skipping /dev/nvidia* device regions
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_fp16.h>

__global__ void add_relu_kernel(const half* a, const half* b, half* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2half(fmaxf(__half2float(a[i]) + __half2float(b[i]), 0.0f));
}

struct MappedRegion {
    uintptr_t start, end;
    bool readable;
    bool is_device;  // backed by /dev/nvidia*
};

static std::vector<MappedRegion> get_mapped_regions() {
    std::vector<MappedRegion> regions;
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) return regions;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        uintptr_t start, end;
        char perms[8];
        unsigned long offset, inode;
        unsigned dev_major, dev_minor;
        char pathname[512] = {};
        int n = sscanf(line, "%lx-%lx %4s %lx %x:%x %lu %511s",
                       &start, &end, perms, &offset, &dev_major, &dev_minor, &inode, pathname);
        bool is_dev = (n >= 8 && (strstr(pathname, "/dev/nvidia") || strstr(pathname, "anon_inode")));
        regions.push_back({start, end, perms[0] == 'r', is_dev});
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
    add_relu_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    auto regions = get_mapped_regions();

    // Step 1: Find QMD v5
    uint64_t cbuf0_gpu_va = 0;
    uint32_t cbuf0_size = 0;
    const uint32_t* qmd_ptr = nullptr;

    for (auto& r : regions) {
        if (!r.readable || r.is_device) continue;
        size_t size = r.end - r.start;
        if (size < 384 || size > 256UL * 1024 * 1024) continue;

        const uint32_t* base = (const uint32_t*)r.start;
        size_t n_dwords = size / 4;

        for (size_t i = 0; i + 96 <= n_dwords; i++) {
            const uint32_t* dw = base + i;
            if (((dw[14] >> 20) & 0xF) != 5) continue;       // qmd_major_version=5
            if ((dw[14] & 0xFF) != 0xA4) continue;            // sass_version SM120
            if (((dw[4] >> 23) & 0x7) != 2) continue;         // qmd_type=GRID_CTA

            uint32_t grid_x = dw[39];
            uint32_t block_x = dw[34] & 0xFFFF;
            if (grid_x != (N+255)/256 || block_x != 256) continue;

            uint64_t cb_lo = dw[42];
            uint64_t cb_hi = dw[43] & 0x7FFFF;
            cbuf0_gpu_va = ((cb_hi << 32) | cb_lo) << 6;
            cbuf0_size = ((dw[43] >> 19) & 0x1FFF) << 4;
            qmd_ptr = dw;
            fprintf(stderr, "QMD found: cbuf0 GPU VA=0x%lx size=%u\n",
                    (unsigned long)cbuf0_gpu_va, cbuf0_size);
            goto found_qmd;
        }
    }
    fprintf(stderr, "No QMD found!\n");
    return 1;

found_qmd:
    // Step 2: Search for cbuf0 in readable non-device regions
    // Match by kernel args at offset 0x380 (DW[224+])
    for (auto& r : regions) {
        if (!r.readable || r.is_device) continue;
        size_t size = r.end - r.start;
        if (size < 928) continue;
        if (size > 64UL * 1024 * 1024) continue;

        const uint32_t* base = (const uint32_t*)r.start;
        size_t n_dwords = size / 4;

        for (size_t i = 0; i + 232 <= n_dwords; i++) {
            const uint32_t* dw = base + i;

            // Check kernel args: a_ptr at DW[224-225], n at DW[230]
            uint64_t arg_a = (uint64_t)dw[225] << 32 | dw[224];
            if (arg_a != (uint64_t)d_a) continue;
            int32_t arg_n = (int32_t)dw[230];
            if (arg_n != N) continue;

            // Found it! Dump everything
            fprintf(stderr, "\n*** CBUF0 FOUND at CPU addr %p ***\n", (void*)dw);
            fprintf(stderr, "  Region: 0x%lx-0x%lx (offset=%zu)\n",
                    r.start, r.end, (uintptr_t)dw - r.start);

            // Verify other args
            uint64_t arg_b = (uint64_t)dw[227] << 32 | dw[226];
            uint64_t arg_y = (uint64_t)dw[229] << 32 | dw[228];
            fprintf(stderr, "  Args: a=%p b=%p y=%p n=%d\n",
                    (void*)arg_a, (void*)arg_b, (void*)arg_y, arg_n);

            // Dump all non-zero driver parameter DWORDs (0-223)
            fprintf(stderr, "\n=== CBUF0 DRIVER PARAMS ===\n");
            for (int j = 0; j < 224; j += 4) {
                if (dw[j] || dw[j+1] || dw[j+2] || dw[j+3]) {
                    fprintf(stderr, "  DW[%3d-%3d] (0x%03x): %08x %08x %08x %08x\n",
                            j, j+3, j*4, dw[j], dw[j+1], dw[j+2], dw[j+3]);
                }
            }

            fprintf(stderr, "\n=== KEY OFFSETS ===\n");
            fprintf(stderr, "  [0x358] DW[214-215] desc:        0x%08x_%08x\n", dw[215], dw[214]);
            fprintf(stderr, "  [0x2f0] DW[188-189] shared_win:  0x%08x_%08x\n", dw[189], dw[188]);
            fprintf(stderr, "  [0x2f8] DW[190-191] local_win:   0x%08x_%08x\n", dw[191], dw[190]);
            fprintf(stderr, "  [0x360] DW[216-218] blockDim:    %u %u %u\n", dw[216], dw[217], dw[218]);
            fprintf(stderr, "  [0x37c] DW[223]:                 0x%08x\n", dw[223]);
            goto done;
        }
    }
    fprintf(stderr, "\nCould not find cbuf0 content in process memory!\n");
    fprintf(stderr, "The CUDA driver may use non-readable GPU-backed memory for cbuf0.\n");
done:
    delete[] h_a;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
    return 0;
}
