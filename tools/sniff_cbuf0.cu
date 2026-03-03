// sniff_cbuf0.cu — Launch add_relu via CUDA, then scan process memory for QMD v5 + cbuf0
//
// Strategy:
//   1. Launch add_relu_kernel with known args via CUDA runtime
//   2. After sync, scan all readable mmap'd regions for QMD v5 signature
//   3. From QMD, extract cbuf0 GPU VA and size
//   4. Scan all regions for cbuf0 data (match known shared_mem_window value)
//   5. Dump the full cbuf0 driver parameter area
//
// The QMD v5 signature: DW[14] bits 23:20 = 5 (qmd_major_version)
// Plus DW[14] bits 15:8 = 0xA4 (sass_version for SM120)

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

// QMD v5 field extraction helpers (bit-level access)
static uint32_t qmd_get(const uint32_t* dw, int hi_bit, int lo_bit) {
    int dw_idx = lo_bit / 32;
    int lo = lo_bit % 32;
    int hi = hi_bit % 32;
    uint32_t mask = ((2u << (hi - lo)) - 1) << lo;
    return (dw[dw_idx] & mask) >> lo;
}

struct MappedRegion {
    uintptr_t start;
    uintptr_t end;
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
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) == 3) {
            regions.push_back({start, end, perms[0] == 'r'});
        }
    }
    fclose(f);
    return regions;
}

struct QMDMatch {
    uintptr_t addr;
    uint64_t cbuf0_gpu_va;
    uint32_t cbuf0_size;
    uint32_t grid_x, block_x;
    uint32_t reg_count;
    uint64_t program_addr;
};

int main() {
    // Launch kernel with CUDA to populate QMD + cbuf0
    const int N = 4096;
    half *d_a, *d_b, *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));

    // Use known values we can search for
    half* h_a = new half[N];
    srand(42);
    for (int i = 0; i < N; i++)
        h_a[i] = __float2half(((float)rand() / RAND_MAX) * 4.0f - 2.0f);
    cudaMemcpy(d_a, h_a, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_a, N * sizeof(half), cudaMemcpyHostToDevice);

    fprintf(stderr, "GPU pointers: a=%p b=%p y=%p  n=%d\n", d_a, d_b, d_y, N);
    fprintf(stderr, "Grid=(%d,1,1) Block=(256,1,1)\n", (N+255)/256);

    add_relu_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    fprintf(stderr, "\n=== Scanning process memory for QMD v5 ===\n");
    auto regions = get_mapped_regions();
    fprintf(stderr, "Found %zu mapped regions\n", regions.size());

    std::vector<QMDMatch> matches;

    for (auto& r : regions) {
        if (!r.readable) continue;
        size_t size = r.end - r.start;
        if (size < 384) continue;  // QMD v5 is 384 bytes
        if (size > 256 * 1024 * 1024) continue;  // skip huge regions

        const uint32_t* base = (const uint32_t*)r.start;
        size_t n_dwords = size / 4;

        // Scan for QMD v5 signature: qmd_major_version=5 at bits 471:468
        // That's DW[14] bits 23:20
        for (size_t i = 0; i + 96 <= n_dwords; i++) {
            const uint32_t* dw = base + i;

            // Check qmd_major_version = 5
            uint32_t major = (dw[14] >> 20) & 0xF;
            if (major != 5) continue;

            // Check sass_version = 0xA4 (SM120) at bits 455:448 = DW[14] bits 7:0
            uint32_t sass = dw[14] & 0xFF;
            if (sass != 0xA4) continue;

            // Check qmd_type = 2 at bits 153:151 = DW[4] bits 25:23
            uint32_t qtype = (dw[4] >> 23) & 0x7;
            if (qtype != 2) continue;

            // Check barrier_count = 1 at bits 1141:1137 = DW[35] bits 21:17
            uint32_t barriers = (dw[35] >> 17) & 0x1F;

            // Extract grid and block dims
            uint32_t grid_x = dw[39];  // bits 1279:1248
            uint32_t block_x = dw[34] & 0xFFFF;  // bits 1103:1088 = DW[34] bits 15:0

            // Extract register count: bits 1136:1128 = DW[35] bits 16:8
            uint32_t regs = (dw[35] >> 8) & 0x1FF;

            // Extract program address (shifted4): bits 1055:1024 = DW[32], 1076:1056 = DW[33] bits 20:0
            uint64_t prog_lo = dw[32];
            uint64_t prog_hi = dw[33] & 0x1FFFFF;
            uint64_t prog_addr = ((prog_hi << 32) | prog_lo) << 4;

            // Extract cbuf0 address (shifted6): base = 1344 + 0*64 = 1344
            // DW[42] bits 31:0 = addr_lower_shifted6, DW[43] bits 18:0 = addr_upper_shifted6
            uint64_t cb_lo = dw[42];
            uint64_t cb_hi = dw[43] & 0x7FFFF;
            uint64_t cbuf0_addr = ((cb_hi << 32) | cb_lo) << 6;

            // Extract cbuf0 size (shifted4): bits 1407:1395 = DW[43] bits 31:19
            uint32_t cb_size_s4 = (dw[43] >> 19) & 0x1FFF;
            uint32_t cb_size = cb_size_s4 << 4;

            // Sanity checks
            if (grid_x == 0 || block_x == 0) continue;
            if (regs == 0 || regs > 255) continue;
            if (cbuf0_addr == 0) continue;

            // Check if this looks like our kernel launch
            if (grid_x == (N+255)/256 && block_x == 256) {
                QMDMatch m;
                m.addr = (uintptr_t)(dw);
                m.cbuf0_gpu_va = cbuf0_addr;
                m.cbuf0_size = cb_size;
                m.grid_x = grid_x;
                m.block_x = block_x;
                m.reg_count = regs;
                m.program_addr = prog_addr;
                matches.push_back(m);

                fprintf(stderr, "\n*** QMD v5 MATCH at %p ***\n", (void*)m.addr);
                fprintf(stderr, "  grid=(%u,_,_) block=(%u,_,_) regs=%u barriers=%u\n",
                        grid_x, block_x, regs, barriers);
                fprintf(stderr, "  program_addr=0x%lx\n", (unsigned long)prog_addr);
                fprintf(stderr, "  cbuf0: gpu_va=0x%lx size=%u (size_shifted4=%u)\n",
                        (unsigned long)cbuf0_addr, cb_size, cb_size_s4);

                // Dump raw QMD DWORDs
                fprintf(stderr, "\n  Raw QMD (96 DWORDs):\n");
                for (int j = 0; j < 96; j += 4) {
                    fprintf(stderr, "    DW[%2d-%2d]: %08x %08x %08x %08x\n",
                            j, j+3, dw[j], dw[j+1], dw[j+2], dw[j+3]);
                }
            }
        }
    }

    if (matches.empty()) {
        fprintf(stderr, "\nNo QMD v5 matches found! The CUDA driver may have reused the memory.\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        delete[] h_a;
        return 1;
    }

    // Now find cbuf0 in mapped memory by scanning for its GPU VA -> CPU mapping
    // We know cbuf0 contains shared_mem_window=0x729400000000 at DWORDs 188-189
    // and 0xfffdc0 at DWORD 223
    fprintf(stderr, "\n=== Scanning for cbuf0 content (shared_window=0x729400000000) ===\n");

    for (auto& m : matches) {
        fprintf(stderr, "\nSearching for cbuf0 gpu_va=0x%lx (size=%u)...\n",
                (unsigned long)m.cbuf0_gpu_va, m.cbuf0_size);

        for (auto& r : regions) {
            if (!r.readable) continue;
            size_t size = r.end - r.start;
            if (size < 924) continue;
            if (size > 64 * 1024 * 1024) continue;  // smaller limit to avoid bus errors
            // Skip kernel/device regions that might bus error
            if (r.start < 0x10000) continue;

            const uint32_t* base = (const uint32_t*)r.start;
            size_t n_dwords = size / 4;

            for (size_t i = 0; i + 232 <= n_dwords; i++) {
                const uint32_t* dw = base + i;

                // Check for shared_mem_window at DW[188-189] = 0x729400000000
                uint64_t shared_win = (uint64_t)dw[189] << 32 | dw[188];
                if (shared_win != 0x729400000000ULL) continue;

                // Double check: DW[223] should be 0xfffdc0
                // (might not be exact, let's be flexible)
                uint32_t magic = dw[223];

                // Check for our kernel args at offset 0x380 (DW[224])
                uint64_t arg_a = (uint64_t)dw[225] << 32 | dw[224];
                uint64_t arg_b = (uint64_t)dw[227] << 32 | dw[226];
                uint64_t arg_y = (uint64_t)dw[229] << 32 | dw[228];
                int32_t  arg_n = (int32_t)dw[230];

                // Check if args match our kernel launch
                if (arg_a == (uint64_t)d_a && arg_n == N) {
                    fprintf(stderr, "\n*** CBUF0 FOUND at %p (offset %zu in region) ***\n",
                            (void*)(dw), (uintptr_t)dw - r.start);
                    fprintf(stderr, "  Kernel args at 0x380:\n");
                    fprintf(stderr, "    a_ptr = 0x%lx (expected %p)\n", (unsigned long)arg_a, d_a);
                    fprintf(stderr, "    b_ptr = 0x%lx (expected %p)\n", (unsigned long)arg_b, d_b);
                    fprintf(stderr, "    y_ptr = 0x%lx (expected %p)\n", (unsigned long)arg_y, d_y);
                    fprintf(stderr, "    n     = %d (expected %d)\n", arg_n, N);
                    fprintf(stderr, "    DW[223] (0x37C) = 0x%08x\n", magic);

                    // Dump the FULL driver parameter area (DWORDs 0-223)
                    fprintf(stderr, "\n  === FULL CBUF0 DRIVER PARAMS (DWORDs 0-223) ===\n");
                    for (int j = 0; j < 224; j += 4) {
                        // Only print non-zero lines to keep output manageable
                        if (dw[j] || dw[j+1] || dw[j+2] || dw[j+3]) {
                            fprintf(stderr, "    DW[%3d-%3d] (0x%03x): %08x %08x %08x %08x\n",
                                    j, j+3, j*4, dw[j], dw[j+1], dw[j+2], dw[j+3]);
                        }
                    }

                    // Specifically dump the interesting offsets
                    fprintf(stderr, "\n  === KEY OFFSETS ===\n");
                    fprintf(stderr, "  [0x358] DW[214-215] descriptor:     0x%08x_%08x\n", dw[215], dw[214]);
                    fprintf(stderr, "  [0x360] DW[216]     blockDim.x?:    0x%08x\n", dw[216]);
                    fprintf(stderr, "  [0x364] DW[217]     blockDim.y?:    0x%08x\n", dw[217]);
                    fprintf(stderr, "  [0x368] DW[218]     blockDim.z?:    0x%08x\n", dw[218]);
                    fprintf(stderr, "  [0x36c] DW[219]:                    0x%08x\n", dw[219]);
                    fprintf(stderr, "  [0x370] DW[220-221]:                0x%08x_%08x\n", dw[221], dw[220]);
                    fprintf(stderr, "  [0x378] DW[222]:                    0x%08x\n", dw[222]);
                    fprintf(stderr, "  [0x37c] DW[223]     magic:          0x%08x\n", dw[223]);

                    fprintf(stderr, "  [0x2f0] DW[188-189] shared_window:  0x%08x_%08x\n", dw[189], dw[188]);
                    fprintf(stderr, "  [0x2f8] DW[190-191] local_window:   0x%08x_%08x\n", dw[191], dw[190]);

                    // Also dump 0x300-0x357 range
                    fprintf(stderr, "\n  [0x300-0x357] range:\n");
                    for (int j = 192; j < 214; j += 4) {
                        if (dw[j] || dw[j+1] || dw[j+2] || dw[j+3]) {
                            fprintf(stderr, "    DW[%3d-%3d] (0x%03x): %08x %08x %08x %08x\n",
                                    j, j+3, j*4, dw[j], dw[j+1], dw[j+2], dw[j+3]);
                        }
                    }

                    // Also check first 48 bytes (pre-Blackwell driver params area)
                    fprintf(stderr, "\n  [0x000-0x02F] first 12 DWORDs:\n");
                    for (int j = 0; j < 12; j += 4) {
                        fprintf(stderr, "    DW[%3d-%3d] (0x%03x): %08x %08x %08x %08x\n",
                                j, j+3, j*4, dw[j], dw[j+1], dw[j+2], dw[j+3]);
                    }

                    goto found;
                }
            }
        }
    }
    fprintf(stderr, "\nCould not find cbuf0 in mapped memory!\n");
found:

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
    delete[] h_a;
    return 0;
}
