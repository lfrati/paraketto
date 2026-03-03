// sniff_cutlass.cu — Launch each CUTLASS GEMM variant via CUDA, dump Params from cbuf0
//
// Strategy: launch each kernel, scan /proc/self/maps for QMD v5 + cbuf0,
// dump the kernel args region (cbuf0[param_base..cbuf0_size]).
// Known pointer values let us identify field offsets.

#include "cutlass_gemm.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

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
        uintptr_t s, e;
        char perms[8];
        if (sscanf(line, "%lx-%lx %4s", &s, &e, perms) == 3)
            regions.push_back({s, e, perms[0] == 'r'});
    }
    fclose(f);
    return regions;
}

// Scan memory for QMD v5 signature, return cbuf0 data
struct CbufDump {
    std::vector<uint8_t> data;
    uint32_t param_base;
    uint32_t cbuf0_size;
    uint64_t program_addr;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t reg_count;
    uint32_t shared_mem;
};

static uint32_t qmd_get(const uint32_t* dw, int hi_bit, int lo_bit) {
    int dw_idx = lo_bit / 32;
    int lo = lo_bit % 32;
    int hi = hi_bit % 32;
    uint32_t mask = ((2u << (hi - lo)) - 1) << lo;
    return (dw[dw_idx] & mask) >> lo;
}

static CbufDump scan_for_latest_qmd() {
    CbufDump result = {};
    auto regions = get_mapped_regions();

    // Look for QMD v5: DW[14] sass_version=0xa4, major=5
    uint64_t best_cbuf0_va = 0;
    const uint32_t* best_qmd = nullptr;

    for (auto& r : regions) {
        if (!r.readable || r.end - r.start < 384) continue;
        const uint8_t* base = (const uint8_t*)r.start;
        size_t len = r.end - r.start;
        for (size_t off = 0; off + 384 <= len; off += 4) {
            const uint32_t* dw = (const uint32_t*)(base + off);
            if ((dw[14] & 0xFFF0FF00) != 0x2F5003A4) continue; // sass + major=5
            uint64_t cbuf0_va = ((uint64_t)(dw[43] & 0x7FFFF) << 32) | dw[42];
            cbuf0_va <<= 6;
            if (cbuf0_va > best_cbuf0_va) {
                best_cbuf0_va = cbuf0_va;
                best_qmd = dw;
            }
        }
    }

    if (!best_qmd) return result;

    // Extract QMD fields
    result.program_addr = ((uint64_t)(best_qmd[33] & 0x1FFFFF) << 32 | best_qmd[32]) << 4;
    result.grid_x = best_qmd[39];
    result.grid_y = best_qmd[40];
    result.grid_z = best_qmd[41];
    result.block_x = best_qmd[34] & 0xFFFF;
    result.block_y = (best_qmd[34] >> 16) & 0xFFFF;
    result.block_z = best_qmd[35] & 0xFF;
    result.reg_count = (best_qmd[35] >> 8) & 0x1FF;
    result.shared_mem = (best_qmd[36] & 0x7FF) << 7;

    uint64_t cbuf0_va = best_cbuf0_va;
    uint32_t cbuf0_size = ((best_qmd[43] >> 19) & 0x1FFF) << 4;
    result.cbuf0_size = cbuf0_size;

    // Find cbuf0 in memory (match shared_mem_window at dw[188])
    for (auto& r : regions) {
        if (!r.readable || r.end - r.start < cbuf0_size) continue;
        const uint8_t* base = (const uint8_t*)r.start;
        size_t len = r.end - r.start;
        for (size_t off = 0; off + cbuf0_size <= len; off += 64) {
            const uint32_t* dw = (const uint32_t*)(base + off);
            // Check shared_mem_window signature at DW[188-189]
            if (dw[188] == 0x00000000 && dw[189] == 0x00007294) {
                result.data.assign(base + off, base + off + cbuf0_size);
                // Find param_base by looking for known CUTLASS signature patterns
                // Params typically start after the driver template (0x380)
                result.param_base = 0x380;
                return result;
            }
        }
    }
    return result;
}

static void dump_params(const char* name, const CbufDump& dump,
                        uint64_t known_a, uint64_t known_b, uint64_t known_c,
                        int m, int n, int k) {
    printf("\n=== %s ===\n", name);
    printf("  grid=(%u,%u,%u) block=(%u,%u,%u) regs=%u smem=%u\n",
           dump.grid_x, dump.grid_y, dump.grid_z,
           dump.block_x, dump.block_y, dump.block_z,
           dump.reg_count, dump.shared_mem);
    printf("  cbuf0_size=%u  param_base=0x%x  program=0x%lx\n",
           dump.cbuf0_size, dump.param_base, (unsigned long)dump.program_addr);

    if (dump.data.empty()) {
        printf("  WARNING: cbuf0 data not found in memory\n");
        return;
    }

    const uint8_t* params = dump.data.data() + dump.param_base;
    int params_size = dump.cbuf0_size - dump.param_base;
    printf("  params_size=%d bytes\n", params_size);

    // Dump raw params as hex + annotated fields
    printf("  Raw params (hex DWORDs from param_base):\n");
    for (int i = 0; i < params_size && i < 512; i += 4) {
        uint32_t v = *(const uint32_t*)(params + i);
        printf("    [0x%03x] +%3d: 0x%08x", dump.param_base + i, i, v);
        // Check if this looks like part of a pointer
        if (i + 4 < params_size) {
            uint64_t v64 = *(const uint64_t*)(params + i);
            if (v64 == known_a) printf("  ← ptr_A (0x%lx)", (unsigned long)known_a);
            else if (v64 == known_b) printf("  ← ptr_B (0x%lx)", (unsigned long)known_b);
            else if (v64 == known_c) printf("  ← ptr_C (0x%lx)", (unsigned long)known_c);
        }
        if (v == (uint32_t)m) printf("  ← M=%d?", m);
        if (v == (uint32_t)n) printf("  ← N=%d?", n);
        if (v == (uint32_t)k) printf("  ← K=%d?", k);
        // Check for alpha=1.0 in fp16 (0x3C00)
        if (v == 0x3C00) printf("  ← alpha=1.0 (fp16)?");
        if (v == 0x00003C00) printf("  ← alpha=1.0 (fp16)?");
        printf("\n");
    }
}

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cutlass_gemm_init(stream);

    // Use distinctive sizes so we can identify M, N, K in the dump
    int M = 63, N = 128, K = 256;
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * 2));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * 2));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * 2));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * 2));

    // Get GPU pointers as uint64_t for pattern matching
    uint64_t ptr_a = (uint64_t)d_A;
    uint64_t ptr_b = (uint64_t)d_B;
    uint64_t ptr_c = (uint64_t)d_C;
    printf("GPU pointers: A=0x%lx B=0x%lx C=0x%lx\n",
           (unsigned long)ptr_a, (unsigned long)ptr_b, (unsigned long)ptr_c);

    // 1. NN GEMM: Y[M,N] = X[M,K] @ W[K,N]
    printf("\n--- Testing NN GEMM ---\n");
    cutlass_gemm_nn(stream, d_A, M, K, d_B, N, d_C);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto dump_nn = scan_for_latest_qmd();
    // In CUTLASS col-major: col_m=N, col_n=M, col_k=K, A_c=W(d_B), B_c=X(d_A)
    dump_params("NN GEMM (col_m=N=128, col_n=M=63, col_k=K=256)", dump_nn,
                ptr_b, ptr_a, ptr_c, N, M, K);

    // 2. TN GEMM: Y[M,N] = X[M,K] @ W[N,K]^T
    printf("\n--- Testing TN GEMM ---\n");
    cutlass_gemm_nt(stream, d_A, M, K, d_B, N, d_C);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto dump_tn = scan_for_latest_qmd();
    dump_params("TN GEMM (col_m=N=128, col_n=M=63, col_k=K=256)", dump_tn,
                ptr_b, ptr_a, ptr_c, N, M, K);

    // 3. Batched NN
    int batch = 4;
    long long strideA = M * K, strideB = K * N, strideC = M * N;
    half *d_bA, *d_bB, *d_bC;
    CUDA_CHECK(cudaMalloc(&d_bA, batch * M * K * 2));
    CUDA_CHECK(cudaMalloc(&d_bB, batch * K * N * 2));
    CUDA_CHECK(cudaMalloc(&d_bC, batch * M * N * 2));
    uint64_t ptr_ba = (uint64_t)d_bA, ptr_bb = (uint64_t)d_bB, ptr_bc = (uint64_t)d_bC;

    printf("\n--- Testing Batched NN GEMM ---\n");
    cutlass_batched_gemm_nn(stream, d_bA, d_bB, d_bC, batch, M, N, K, strideA, strideB, strideC);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto dump_bnn = scan_for_latest_qmd();
    dump_params("Batched NN (batch=4, col_m=N=128, col_n=M=63, col_k=K=256)", dump_bnn,
                ptr_bb, ptr_ba, ptr_bc, N, M, K);

    // 4. Batched TN
    printf("\n--- Testing Batched TN GEMM ---\n");
    cutlass_batched_gemm_nt(stream, d_bA, d_bB, d_bC, batch, M, N, K, strideA, strideB, strideC);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto dump_btn = scan_for_latest_qmd();
    dump_params("Batched TN (batch=4, col_m=N=128, col_n=M=63, col_k=K=256)", dump_btn,
                ptr_bb, ptr_ba, ptr_bc, N, M, K);

    cutlass_gemm_free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    printf("\nDone.\n");
    return 0;
}
