// test_cutlass_cudaless.cpp — Test CUTLASS GEMM via cudaless ioctls
//
// Tests 4 main variants: gemm_nn, gemm_nt, batched_nn, batched_nt
// Each test: generate random FP16 matrices, launch via gpu.h, compare to CPU reference.
// Tolerance: relative 2% for FP16 tensor core accumulation.

#include "cutlass_cudaless.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>

// =========================================================================
// FP16 helpers
// =========================================================================

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (x >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | frac;
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) { float f; uint32_t v = sign; memcpy(&f, &v, 4); return f; }
        // Denorm
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        exp++; frac &= 0x3FF;
    } else if (exp == 31) {
        uint32_t v = sign | 0x7F800000 | (frac << 13);
        float f; memcpy(&f, &v, 4); return f;
    }
    uint32_t v = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float f; memcpy(&f, &v, 4);
    return f;
}

static void fill_random_fp16(uint16_t* data, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        float val = ((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f; // [-1, 1]
        data[i] = fp32_to_fp16(val);
    }
}

// =========================================================================
// CPU reference GEMM
// =========================================================================

// C[m,n] = alpha * A[m,k] @ B[k,n] + beta * C[m,n]  (row-major)
static void cpu_gemm_nn(const uint16_t* A, const uint16_t* B, uint16_t* C,
                         int m, int n, int k, float alpha, float beta) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++)
                sum += fp16_to_fp32(A[i * k + p]) * fp16_to_fp32(B[p * n + j]);
            float old = fp16_to_fp32(C[i * n + j]);
            C[i * n + j] = fp32_to_fp16(alpha * sum + beta * old);
        }
    }
}

// C[m,n] = alpha * A[m,k] @ B[n,k]^T + beta * C[m,n]  (row-major)
static void cpu_gemm_nt(const uint16_t* A, const uint16_t* B, uint16_t* C,
                         int m, int n, int k, float alpha, float beta) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++)
                sum += fp16_to_fp32(A[i * k + p]) * fp16_to_fp32(B[j * k + p]);
            float old = fp16_to_fp32(C[i * n + j]);
            C[i * n + j] = fp32_to_fp16(alpha * sum + beta * old);
        }
    }
}

// =========================================================================
// Comparison helper
// =========================================================================

static bool compare_fp16(const uint16_t* gpu, const uint16_t* cpu, int count,
                          float rtol, float atol, const char* name) {
    int errors = 0;
    float max_rel = 0;
    for (int i = 0; i < count; i++) {
        float g = fp16_to_fp32(gpu[i]);
        float c = fp16_to_fp32(cpu[i]);
        float diff = fabsf(g - c);
        float rel = (fabsf(c) > 1e-6f) ? diff / fabsf(c) : diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > atol && rel > rtol) {
            if (errors < 5)
                printf("    MISMATCH [%d]: gpu=%.6f cpu=%.6f diff=%.6f rel=%.4f\n",
                       i, g, c, diff, rel);
            errors++;
        }
    }
    if (errors == 0) {
        printf("  %s: PASS (max_rel=%.4f)\n", name, max_rel);
        return true;
    } else {
        printf("  %s: FAIL (%d/%d errors, max_rel=%.4f)\n", name, errors, count, max_rel);
        return false;
    }
}

// =========================================================================
// Test functions
// =========================================================================

static bool test_gemm_nn(CutlassCudaless& cl) {
    const int M = 64, N = 64, K = 64;

    // Diagnostic 1: zero-input test
    {
        auto a0 = cl.gpu->gpu_malloc(M * K * 2);
        auto b0 = cl.gpu->gpu_malloc(K * N * 2);
        auto c0 = cl.gpu->gpu_malloc(M * N * 2);
        memset(a0.cpu_ptr, 0, M * K * 2);
        memset(b0.cpu_ptr, 0, K * N * 2);
        for (int i = 0; i < M * N; i++) ((uint16_t*)c0.cpu_ptr)[i] = 0x3C00;
        __sync_synchronize();
        cl.gpu->begin_commands();
        cl.gemm_nn(a0.gpu_addr, M, K, b0.gpu_addr, N, c0.gpu_addr);
        cl.gpu->wait_kernel();
        __sync_synchronize();
        uint16_t* out = (uint16_t*)c0.cpu_ptr;
        int nonzero = 0;
        for (int i = 0; i < M * N; i++) if (out[i] != 0) nonzero++;
        printf("  [diag] zero-input test: %d/%d non-zero outputs (should be 0)\n", nonzero, M*N);
    }

    // Diagnostic 2: identity matrix test (A=I, B=sequential => C should = B)
    {
        auto ai = cl.gpu->gpu_malloc(M * K * 2);
        auto bi = cl.gpu->gpu_malloc(K * N * 2);
        auto ci = cl.gpu->gpu_malloc(M * N * 2);
        uint16_t* hA = (uint16_t*)ai.cpu_ptr;
        uint16_t* hB = (uint16_t*)bi.cpu_ptr;
        uint16_t* hC = (uint16_t*)ci.cpu_ptr;
        memset(hA, 0, M * K * 2);
        for (int i = 0; i < M && i < K; i++) hA[i * K + i] = fp32_to_fp16(1.0f);
        // B = sequential: B[i,j] = (i*N + j + 1) * 0.01
        for (int i = 0; i < K; i++)
            for (int j = 0; j < N; j++)
                hB[i * N + j] = fp32_to_fp16((float)(i * N + j + 1) * 0.01f);
        memset(hC, 0, M * N * 2);
        __sync_synchronize();
        cl.gpu->begin_commands();
        cl.gemm_nn(ai.gpu_addr, M, K, bi.gpu_addr, N, ci.gpu_addr);
        cl.gpu->wait_kernel();
        __sync_synchronize();
        // Expected: C = I @ B = B
        printf("  [diag] identity test C=I@B (should equal B):\n");
        int id_err = 0;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float gpu = fp16_to_fp32(hC[i * N + j]);
                float exp = fp16_to_fp32(hB[i * N + j]);
                if (fabsf(gpu - exp) > 0.01f) {
                    if (id_err < 10)
                        printf("    C[%d,%d] gpu=%.6f expected=%.6f\n", i, j, gpu, exp);
                    id_err++;
                }
            }
        if (id_err == 0)
            printf("    PASS: C == B (identity works with non-uniform data!)\n");
        else {
            printf("    FAIL: %d/%d mismatches\n", id_err, M*N);
            // Show first row of GPU output vs expected
            printf("    GPU row 0:");
            for (int j = 0; j < 8; j++) printf(" %.4f", fp16_to_fp32(hC[j]));
            printf("\n    Exp row 0:");
            for (int j = 0; j < 8; j++) printf(" %.4f", fp16_to_fp32(hB[j]));
            printf("\n");
            // Check if GPU output = B^T (transposed)
            int t_err = 0;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float gpu = fp16_to_fp32(hC[i * N + j]);
                    float bt = fp16_to_fp32(hB[j * N + i]);  // B^T
                    if (fabsf(gpu - bt) > 0.01f) t_err++;
                }
            printf("    Transposed check: %d/%d mismatches (0 = output is B^T)\n", t_err, M*N);
        }
    }

    // Diagnostic 3: all-ones test (each element=1.0, result should be K=64.0)
    {
        auto a1 = cl.gpu->gpu_malloc(M * K * 2);
        auto b1 = cl.gpu->gpu_malloc(K * N * 2);
        auto c1 = cl.gpu->gpu_malloc(M * N * 2);
        for (int i = 0; i < M * K; i++) ((uint16_t*)a1.cpu_ptr)[i] = 0x3C00; // 1.0
        for (int i = 0; i < K * N; i++) ((uint16_t*)b1.cpu_ptr)[i] = 0x3C00; // 1.0
        memset(c1.cpu_ptr, 0, M * N * 2);
        __sync_synchronize();
        printf("  [diag] ones test: A@0x%lx B@0x%lx C@0x%lx\n",
               (unsigned long)a1.gpu_addr, (unsigned long)b1.gpu_addr, (unsigned long)c1.gpu_addr);
        cl.gpu->begin_commands();
        cl.gemm_nn(a1.gpu_addr, M, K, b1.gpu_addr, N, c1.gpu_addr);
        cl.gpu->wait_kernel();
        __sync_synchronize();
        uint16_t* out = (uint16_t*)c1.cpu_ptr;
        printf("  [diag] ones test first 8:");
        for (int i = 0; i < 8; i++) printf(" %.1f", fp16_to_fp32(out[i]));
        printf(" (expected all 64.0)\n");
        // Check if ALL values are the same
        int same = 0;
        for (int i = 1; i < M * N; i++) if (out[i] == out[0]) same++;
        printf("  [diag] ones test: %d/%d elements equal to first (%.1f)\n",
               same+1, M*N, fp16_to_fp32(out[0]));
    }

    // Allocate GPU memory
    auto alloc_A = cl.gpu->gpu_malloc(M * K * 2);
    auto alloc_B = cl.gpu->gpu_malloc(K * N * 2);
    auto alloc_C = cl.gpu->gpu_malloc(M * N * 2);

    // Generate random data
    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, M * K, 42);
    fill_random_fp16(h_B, K * N, 123);
    memset(h_C, 0, M * N * 2);
    __sync_synchronize();

    // CPU reference
    uint16_t* cpu_C = new uint16_t[M * N]();
    cpu_gemm_nn(h_A, h_B, cpu_C, M, N, K, 1.0f, 0.0f);

    // GPU launch
    cl.gpu->begin_commands();
    cl.gemm_nn(alloc_A.gpu_addr, M, K, alloc_B.gpu_addr, N, alloc_C.gpu_addr);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    // Verify input data readback
    {
        uint16_t v0 = h_A[0], v1 = h_A[1], v100 = h_A[100];
        printf("  [diag] readback A: [0]=0x%04x(%.4f) [1]=0x%04x(%.4f) [100]=0x%04x(%.4f)\n",
               v0, fp16_to_fp32(v0), v1, fp16_to_fp32(v1), v100, fp16_to_fp32(v100));
        // Re-generate to verify
        uint16_t check[4096];
        fill_random_fp16(check, M*K, 42);
        if (memcmp(h_A, check, M*K*2) != 0) printf("  [diag] WARNING: A readback mismatch!\n");
        else printf("  [diag] A readback matches expected data\n");
        fill_random_fp16(check, K*N, 123);
        if (memcmp(h_B, check, K*N*2) != 0) printf("  [diag] WARNING: B readback mismatch!\n");
        else printf("  [diag] B readback matches expected data\n");
    }

    // Dump first 16 GPU values vs CPU
    printf("  [diag] GPU vs CPU first 16:\n");
    for (int i = 0; i < 16; i++)
        printf("    [%2d] gpu=%.6f cpu=%.6f\n", i, fp16_to_fp32(h_C[i]), fp16_to_fp32(cpu_C[i]));

    // Diagnostic: compute h_B @ h_A to check if matrices are swapped
    uint16_t* swap_C = new uint16_t[M * N]();
    cpu_gemm_nn(h_B, h_A, swap_C, N, M, K, 1.0f, 0.0f);  // B@A instead of A@B
    printf("  [diag] Checking if GPU = B@A (swapped):\n");
    for (int i = 0; i < 8; i++)
        printf("    [%2d] gpu=%.6f A@B=%.6f B@A=%.6f\n", i,
               fp16_to_fp32(h_C[i]), fp16_to_fp32(cpu_C[i]), fp16_to_fp32(swap_C[i]));
    delete[] swap_C;

    // Diagnostic: try also interpreting output as transposed
    printf("  [diag] Checking if GPU output = (A@B)^T (col-major read):\n");
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) {
            int row_idx = i * N + j;
            int col_idx = j * M + i;  // transposed index
            printf("    C[%d,%d] gpu=%.6f cpu_row[%d,%d]=%.6f cpu_col[%d,%d]=%.6f\n",
                   i, j, fp16_to_fp32(h_C[row_idx]),
                   i, j, fp16_to_fp32(cpu_C[row_idx]),
                   j, i, fp16_to_fp32(cpu_C[col_idx]));
        }

    bool ok = compare_fp16(h_C, cpu_C, M * N, 0.02f, 0.01f, "gemm_nn 64x64x64");
    delete[] cpu_C;
    return ok;
}

static bool test_gemm_nt(CutlassCudaless& cl) {
    const int M = 64, N = 128, K = 64;

    auto alloc_A = cl.gpu->gpu_malloc(M * K * 2);
    auto alloc_B = cl.gpu->gpu_malloc(N * K * 2);  // B is [N, K] for NT
    auto alloc_C = cl.gpu->gpu_malloc(M * N * 2);

    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, M * K, 77);
    fill_random_fp16(h_B, N * K, 88);
    memset(h_C, 0, M * N * 2);
    __sync_synchronize();

    uint16_t* cpu_C = new uint16_t[M * N]();
    cpu_gemm_nt(h_A, h_B, cpu_C, M, N, K, 1.0f, 0.0f);

    cl.gpu->begin_commands();
    cl.gemm_nt(alloc_A.gpu_addr, M, K, alloc_B.gpu_addr, N, alloc_C.gpu_addr);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    bool ok = compare_fp16(h_C, cpu_C, M * N, 0.02f, 0.01f, "gemm_nt 64x128x64");
    delete[] cpu_C;
    return ok;
}

static bool test_batched_nn(CutlassCudaless& cl) {
    const int batch = 4, M = 64, N = 64, K = 64;
    long long sA = M * K, sB = K * N, sC = M * N;

    auto alloc_A = cl.gpu->gpu_malloc(batch * sA * 2);
    auto alloc_B = cl.gpu->gpu_malloc(batch * sB * 2);
    auto alloc_C = cl.gpu->gpu_malloc(batch * sC * 2);

    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, batch * M * K, 111);
    fill_random_fp16(h_B, batch * K * N, 222);
    memset(h_C, 0, batch * sC * 2);
    __sync_synchronize();

    uint16_t* cpu_C = new uint16_t[batch * M * N]();
    for (int b = 0; b < batch; b++)
        cpu_gemm_nn(h_A + b * sA, h_B + b * sB, cpu_C + b * sC, M, N, K, 1.0f, 0.0f);

    cl.gpu->begin_commands();
    cl.batched_gemm_nn(alloc_A.gpu_addr, alloc_B.gpu_addr, alloc_C.gpu_addr,
                       batch, M, N, K, sA, sB, sC);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    bool ok = compare_fp16(h_C, cpu_C, batch * M * N, 0.02f, 0.01f,
                           "batched_nn 4x64x64x64");
    delete[] cpu_C;
    return ok;
}

static bool test_batched_nt(CutlassCudaless& cl) {
    const int batch = 4, M = 64, N = 64, K = 64;
    long long sA = M * K, sB = N * K, sC = M * N;

    auto alloc_A = cl.gpu->gpu_malloc(batch * sA * 2);
    auto alloc_B = cl.gpu->gpu_malloc(batch * sB * 2);
    auto alloc_C = cl.gpu->gpu_malloc(batch * sC * 2);

    uint16_t* h_A = (uint16_t*)alloc_A.cpu_ptr;
    uint16_t* h_B = (uint16_t*)alloc_B.cpu_ptr;
    uint16_t* h_C = (uint16_t*)alloc_C.cpu_ptr;
    fill_random_fp16(h_A, batch * M * K, 333);
    fill_random_fp16(h_B, batch * N * K, 444);
    memset(h_C, 0, batch * sC * 2);
    __sync_synchronize();

    uint16_t* cpu_C = new uint16_t[batch * M * N]();
    for (int b = 0; b < batch; b++)
        cpu_gemm_nt(h_A + b * sA, h_B + b * sB, cpu_C + b * sC, M, N, K, 1.0f, 0.0f);

    cl.gpu->begin_commands();
    cl.batched_gemm_nt(alloc_A.gpu_addr, alloc_B.gpu_addr, alloc_C.gpu_addr,
                       batch, M, N, K, sA, sB, sC);
    cl.gpu->wait_kernel();
    __sync_synchronize();

    bool ok = compare_fp16(h_C, cpu_C, batch * M * N, 0.02f, 0.01f,
                           "batched_nt 4x64x64x64");
    delete[] cpu_C;
    return ok;
}

// =========================================================================
// Main
// =========================================================================

int main() {
    printf("=== CUTLASS Cudaless Test ===\n\n");

    GPU gpu;
    gpu.init();

    CutlassCudaless cl;
    if (!cl.init(gpu, "cutlass_gemm.cubin")) {
        fprintf(stderr, "Failed to init CutlassCudaless\n");
        return 1;
    }
    printf("Loaded %zu CUTLASS kernels\n\n", cl.cubin.kernels.size());

    int pass = 0, total = 0;

    total++; if (test_gemm_nn(cl)) pass++;
    total++; if (test_gemm_nt(cl)) pass++;
    total++; if (test_batched_nn(cl)) pass++;
    total++; if (test_batched_nt(cl)) pass++;

    printf("\n=== Results: %d/%d PASS ===\n", pass, total);
    return (pass == total) ? 0 : 1;
}
