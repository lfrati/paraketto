// cutlass_cudaless.h — CUTLASS GEMM via direct ioctls (no libcuda.so)
//
// Loads cutlass_gemm.cubin, identifies all 10 kernel variants by mangled name
// features, and provides simple C++ launch functions that use gpu.h.
//
// Params construction is done by the cutlass_params.o bridge (compiled with nvcc).
// Kernel launch is done via gpu.h's launch_kernel() (pure ioctls).

#pragma once

#include "gpu.h"
#include "cubin_loader.h"
#include <cstdint>
#include <cstring>

// =========================================================================
// extern "C" declarations for the params bridge (cutlass_params.o)
// =========================================================================

#define GEMM_PARAMS_DECL(name) \
extern "C" int name(void* out, int out_sz, int* grid, int* block, int* smem, \
    int M, int N, int K, uint64_t A, int ldA, uint64_t B, int ldB, \
    uint64_t C, int ldC, uint16_t alpha, uint16_t beta);

#define BATCH_PARAMS_DECL(name) \
extern "C" int name(void* out, int out_sz, int* grid, int* block, int* smem, \
    int M, int N, int K, uint64_t A, int ldA, long long sA, \
    uint64_t B, int ldB, long long sB, uint64_t C, int ldC, long long sC, \
    int batch, uint16_t alpha, uint16_t beta);

GEMM_PARAMS_DECL(cutlass_params_nn_64x64_64_s6)
GEMM_PARAMS_DECL(cutlass_params_nn_64x64_32_s10)
GEMM_PARAMS_DECL(cutlass_params_nn_64x64_32_s3)
GEMM_PARAMS_DECL(cutlass_params_nn_64x64_32_s6_a2)
GEMM_PARAMS_DECL(cutlass_params_nn_128x128_32_s5)
GEMM_PARAMS_DECL(cutlass_params_tn_64x64_64_s6)
BATCH_PARAMS_DECL(cutlass_params_batched_nn_64x64_64_s6)
BATCH_PARAMS_DECL(cutlass_params_batched_nn_64x64_32_s2_a8b1)
BATCH_PARAMS_DECL(cutlass_params_batched_tn_64x64_64_s6)
BATCH_PARAMS_DECL(cutlass_params_batched_tn_64x64_64_s6_e1)

#undef GEMM_PARAMS_DECL
#undef BATCH_PARAMS_DECL

// =========================================================================
// Kernel variant enum
// =========================================================================

enum CutlassVariant {
    CUTLASS_NN_64x64_64_s6 = 0,
    CUTLASS_NN_64x64_32_s10,
    CUTLASS_NN_64x64_32_s3,
    CUTLASS_NN_64x64_32_s6_a2,
    CUTLASS_NN_128x128_32_s5,
    CUTLASS_TN_64x64_64_s6,
    CUTLASS_BATCH_NN_64x64_64_s6,
    CUTLASS_BATCH_NN_64x64_32_s2,
    CUTLASS_BATCH_TN_64x64_64_s6,
    CUTLASS_BATCH_TN_64x64_64_s6_e1,
    CUTLASS_NUM_VARIANTS
};

// =========================================================================
// Main wrapper struct
// =========================================================================

struct CutlassCudaless {
    GPU* gpu = nullptr;
    CubinLoader cubin;
    uint64_t cubin_base = 0;
    CubinKernel* kernels[CUTLASS_NUM_VARIANTS] = {};

    bool init(GPU& g, const char* cubin_path) {
        gpu = &g;
        if (!cubin.load(cubin_path)) {
            fprintf(stderr, "CutlassCudaless: failed to load %s\n", cubin_path);
            return false;
        }
        cubin_base = g.upload_cubin(cubin.image.data(), cubin.image.size());
        if (!cubin_base) {
            fprintf(stderr, "CutlassCudaless: upload_cubin failed\n");
            return false;
        }
        fprintf(stderr, "CutlassCudaless: cubin_base=0x%lx size=%zu\n",
                (unsigned long)cubin_base, cubin.image.size());
        return identify_kernels();
    }

    // FP16 constants
    static uint16_t fp16_one()  { return 0x3C00; }
    static uint16_t fp16_zero() { return 0x0000; }

    // Check if all leading dims and strides are 8-element aligned (16 bytes for FP16)
    static bool batched_aligned(int ld1, int ld2, int ld3,
                                long long s1, long long s2, long long s3) {
        return (ld1 % 8 == 0) && (ld2 % 8 == 0) && (ld3 % 8 == 0) &&
               (s1 % 8 == 0) && (s2 % 8 == 0) && (s3 % 8 == 0);
    }

    // =====================================================================
    // Row-major GEMM API (matches cutlass_gemm.h convention)
    //   NN: Y[m,n] = X[m,k] @ W[k,n]
    //   NT: Y[m,n] = X[m,k] @ W[n,k]^T
    //
    // Internally converts to CUTLASS col-major convention:
    //   col_m=n, col_n=m, col_k=k
    //   CUTLASS_A = W (ptr_B for NN, ptr_B for TN)
    //   CUTLASS_B = X (ptr_A)
    //   CUTLASS_C = Y (ptr_C)
    // =====================================================================

    // Y[m,n] = X[m,k] @ W[k,n]  — NN (both col-major in CUTLASS)
    void gemm_nn(uint64_t X, int m, int k, uint64_t W, int n, uint64_t Y,
                 uint16_t alpha = 0x3C00, uint16_t beta = 0x0000) {
        // Row→Col: CUTLASS A=W(ldA=n), B=X(ldB=k), C=Y(ldC=n), problem=(n,m,k)
        launch_gemm(CUTLASS_NN_64x64_64_s6,
                     cutlass_params_nn_64x64_64_s6,
                     n, m, k, W, n, X, k, Y, n, alpha, beta);
    }

    // Y[m,n] = X[m,k] @ W[n,k]^T — TN in CUTLASS convention
    void gemm_nt(uint64_t X, int m, int k, uint64_t W, int n, uint64_t Y,
                 uint16_t alpha = 0x3C00, uint16_t beta = 0x0000) {
        // Row→Col: CUTLASS A=W^T(ldA=k), B=X(ldB=k), C=Y(ldC=n), problem=(n,m,k)
        launch_gemm(CUTLASS_TN_64x64_64_s6,
                     cutlass_params_tn_64x64_64_s6,
                     n, m, k, W, k, X, k, Y, n, alpha, beta);
    }

    // Y[m,n] += X[m,k] @ W[n,k]^T — NT + accumulate (beta=1)
    void gemm_nt_accum(uint64_t X, int m, int k, uint64_t W, int n, uint64_t Y) {
        launch_gemm(CUTLASS_TN_64x64_64_s6,
                     cutlass_params_tn_64x64_64_s6,
                     n, m, k, W, k, X, k, Y, n, fp16_one(), fp16_one());
    }

    // C[b,m,n] = A[b,m,k] @ B[b,k,n]  — batched NN
    void batched_gemm_nn(uint64_t A, uint64_t B, uint64_t C,
                          int batch, int m, int n, int k,
                          long long strideA, long long strideB, long long strideC) {
        // Row→Col: swap A/B, (n,m,k), ldA=n,ldB=k,ldC=n
        if (batched_aligned(n, k, n, strideB, strideA, strideC))
            launch_batched(CUTLASS_BATCH_NN_64x64_64_s6,
                            cutlass_params_batched_nn_64x64_64_s6,
                            n, m, k, B, n, strideB, A, k, strideA, C, n, strideC,
                            batch, fp16_one(), fp16_zero());
        else
            launch_batched(CUTLASS_BATCH_NN_64x64_32_s2,
                            cutlass_params_batched_nn_64x64_32_s2_a8b1,
                            n, m, k, B, n, strideB, A, k, strideA, C, n, strideC,
                            batch, fp16_one(), fp16_zero());
    }

    // C[b,m,n] = A[b,m,k] @ B[b,n,k]^T — batched NT
    void batched_gemm_nt(uint64_t A, uint64_t B, uint64_t C,
                          int batch, int m, int n, int k,
                          long long strideA, long long strideB, long long strideC) {
        // Row→Col: swap A/B, (n,m,k), ldA=k(transposed B),ldB=k,ldC=n
        if (batched_aligned(k, k, n, strideB, strideA, strideC))
            launch_batched(CUTLASS_BATCH_TN_64x64_64_s6,
                            cutlass_params_batched_tn_64x64_64_s6,
                            n, m, k, B, k, strideB, A, k, strideA, C, n, strideC,
                            batch, fp16_one(), fp16_zero());
        else
            launch_batched(CUTLASS_BATCH_TN_64x64_64_s6_e1,
                            cutlass_params_batched_tn_64x64_64_s6_e1,
                            n, m, k, B, k, strideB, A, k, strideA, C, n, strideC,
                            batch, fp16_one(), fp16_zero());
    }

    // Explicit NN batched: C[b,m,n] = A[b,m,k] @ B[b,k,n]
    void batched_gemm_nn_ex(uint64_t A, int ldA, long long strideA,
                             uint64_t B, int ldB, long long strideB,
                             uint64_t C, int ldC, long long strideC,
                             int batch, int m, int n, int k) {
        // Row→Col: CUTLASS A=B(ldA=ldB), CUTLASS B=A(ldB=ldA), (n,m,k)
        if (batched_aligned(ldB, ldA, ldC, strideB, strideA, strideC))
            launch_batched(CUTLASS_BATCH_NN_64x64_64_s6,
                            cutlass_params_batched_nn_64x64_64_s6,
                            n, m, k, B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
                            batch, fp16_one(), fp16_zero());
        else
            launch_batched(CUTLASS_BATCH_NN_64x64_32_s2,
                            cutlass_params_batched_nn_64x64_32_s2_a8b1,
                            n, m, k, B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
                            batch, fp16_one(), fp16_zero());
    }

    // Explicit NT batched: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
    void batched_gemm_nt_ex(uint64_t A, int ldA, long long strideA,
                             uint64_t B, int ldB, long long strideB,
                             uint64_t C, int ldC, long long strideC,
                             int batch, int m, int n, int k) {
        if (batched_aligned(ldB, ldA, ldC, strideB, strideA, strideC))
            launch_batched(CUTLASS_BATCH_TN_64x64_64_s6,
                            cutlass_params_batched_tn_64x64_64_s6,
                            n, m, k, B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
                            batch, fp16_one(), fp16_zero());
        else
            launch_batched(CUTLASS_BATCH_TN_64x64_64_s6_e1,
                            cutlass_params_batched_tn_64x64_64_s6_e1,
                            n, m, k, B, ldB, strideB, A, ldA, strideA, C, ldC, strideC,
                            batch, fp16_one(), fp16_zero());
    }

private:
    // =====================================================================
    // Kernel identification — map cubin kernels to known variants
    // =====================================================================

    static bool has(const std::string& s, const char* p) {
        return s.find(p) != std::string::npos;
    }
    static int count(const std::string& s, const char* p) {
        int c = 0;
        size_t pos = 0;
        while ((pos = s.find(p, pos)) != std::string::npos) { c++; pos++; }
        return c;
    }

    bool identify_kernels() {
        for (auto& k : cubin.kernels) {
            auto& n = k.mangled_name;
            bool batched = has(n, "GemmBatched");
            bool k64 = has(n, "64ELi64ELi64E");
            bool t128 = has(n, "128ELi128E");
            bool pipe = has(n, "Pipelined");
            bool a2 = has(n, "ISD_Li2ELb0EEELb0E");
            bool e1 = has(n, "LinearCombinationISD_Li1E");
            int nCong = count(n, "Congruous");

            int stages = 0;
            if (has(n, "EELi3ELN")) stages = 3;
            else if (has(n, "EELi5ELN")) stages = 5;
            else if (has(n, "EELi6ELN")) stages = 6;
            else if (has(n, "EELi10ELN")) stages = 10;
            else if (pipe) stages = 2;

            CutlassVariant v = CUTLASS_NUM_VARIANTS;
            if (!batched) {
                if (k64 && nCong == 0)         v = CUTLASS_NN_64x64_64_s6;
                else if (k64 && nCong > 0)     v = CUTLASS_TN_64x64_64_s6;
                else if (t128)                 v = CUTLASS_NN_128x128_32_s5;
                else if (stages == 3)          v = CUTLASS_NN_64x64_32_s3;
                else if (stages == 10)         v = CUTLASS_NN_64x64_32_s10;
                else if (a2)                   v = CUTLASS_NN_64x64_32_s6_a2;
            } else {
                if (k64 && nCong == 0 && !e1)  v = CUTLASS_BATCH_NN_64x64_64_s6;
                else if (k64 && nCong == 0 && e1)  v = CUTLASS_BATCH_TN_64x64_64_s6_e1;
                else if (k64 && nCong > 0)     v = CUTLASS_BATCH_TN_64x64_64_s6;
                else if (pipe)                 v = CUTLASS_BATCH_NN_64x64_32_s2;
            }

            if (v < CUTLASS_NUM_VARIANTS)
                kernels[v] = &k;
        }

        // Verify all variants found
        const char* names[] = {
            "NN_64x64_64_s6", "NN_64x64_32_s10", "NN_64x64_32_s3",
            "NN_64x64_32_s6_a2", "NN_128x128_32_s5", "TN_64x64_64_s6",
            "Batch_NN_64x64_64_s6", "Batch_NN_64x64_32_s2",
            "Batch_TN_64x64_64_s6", "Batch_TN_64x64_64_s6_e1"
        };
        bool ok = true;
        for (int i = 0; i < CUTLASS_NUM_VARIANTS; i++) {
            if (!kernels[i]) {
                fprintf(stderr, "CutlassCudaless: missing kernel variant %s\n", names[i]);
                ok = false;
            } else {
                auto* k = kernels[i];
                fprintf(stderr, "  CUTLASS[%d] %s: regs=%u smem=%u cbuf0=%u param_base=%u "
                        "local_low=%u local_high=%u code_off=0x%lx code_sz=%lu\n",
                        i, names[i], k->reg_count, k->shared_mem_size,
                        k->cbuf0_size, k->param_base,
                        k->local_mem_low, k->local_mem_high,
                        (unsigned long)k->code_offset, (unsigned long)k->code_size);
            }
        }
        return ok;
    }

    // =====================================================================
    // Internal launch helpers
    // =====================================================================

    using GemmParamsFn = int(*)(void*, int, int*, int*, int*,
        int, int, int, uint64_t, int, uint64_t, int, uint64_t, int,
        uint16_t, uint16_t);

    using BatchParamsFn = int(*)(void*, int, int*, int*, int*,
        int, int, int, uint64_t, int, long long,
        uint64_t, int, long long, uint64_t, int, long long,
        int, uint16_t, uint16_t);

    void launch_gemm(CutlassVariant var, GemmParamsFn params_fn,
                     int M, int N, int K,
                     uint64_t A, int ldA, uint64_t B, int ldB,
                     uint64_t C, int ldC,
                     uint16_t alpha, uint16_t beta) {
        CubinKernel* k = kernels[var];
        if (!k) { fprintf(stderr, "CUTLASS kernel %d not found\n", var); return; }

        uint8_t params[512];
        int grid[3], block[3], smem;
        int psz = params_fn(params, 512, grid, block, &smem,
                            M, N, K, A, ldA, B, ldB, C, ldC, alpha, beta);
        if (psz < 0) { fprintf(stderr, "CUTLASS params build failed for variant %d\n", var); return; }

        uint32_t total_smem = k->shared_mem_size + (uint32_t)smem;

        fprintf(stderr, "  [gemm var=%d] M=%d N=%d K=%d A=0x%lx ldA=%d B=0x%lx ldB=%d C=0x%lx ldC=%d\n"
                "    regs=%u smem=%u+%d=%u grid=(%d,%d,%d) block=(%d,%d,%d) psz=%d\n",
                var, M, N, K, (unsigned long)A, ldA, (unsigned long)B, ldB, (unsigned long)C, ldC,
                k->reg_count, k->shared_mem_size, smem, total_smem,
                grid[0], grid[1], grid[2], block[0], block[1], block[2], psz);
        // Dump ALL params as hex, 8 words per line
        for (int row = 0; row < psz; row += 32) {
            fprintf(stderr, "    params[%03x]: ", row);
            for (int i = row; i < row + 32 && i < psz; i += 4)
                fprintf(stderr, "%08x ", *(uint32_t*)(params + i));
            fprintf(stderr, "\n");
        }

        auto cbuf = gpu->prepare_cbuf0(params, psz, k->cbuf0_size, k->param_base,
                                       block[0], block[1], block[2]);

        gpu->launch_kernel(cubin_base + k->code_offset, k->code_size,
                           k->reg_count, total_smem,
                           grid[0], grid[1], grid[2],
                           block[0], block[1], block[2],
                           cbuf.gpu_addr, k->cbuf0_size);
    }

    void launch_batched(CutlassVariant var, BatchParamsFn params_fn,
                        int M, int N, int K,
                        uint64_t A, int ldA, long long sA,
                        uint64_t B, int ldB, long long sB,
                        uint64_t C, int ldC, long long sC,
                        int batch, uint16_t alpha, uint16_t beta) {
        CubinKernel* k = kernels[var];
        if (!k) { fprintf(stderr, "CUTLASS batched kernel %d not found\n", var); return; }

        uint8_t params[512];
        int grid[3], block[3], smem;
        int psz = params_fn(params, 512, grid, block, &smem,
                            M, N, K, A, ldA, sA, B, ldB, sB, C, ldC, sC,
                            batch, alpha, beta);
        if (psz < 0) { fprintf(stderr, "CUTLASS batched params build failed for variant %d\n", var); return; }

        uint32_t total_smem = k->shared_mem_size + (uint32_t)smem;

        fprintf(stderr, "  [batched var=%d] regs=%u smem_static=%u smem_dyn=%d total=%u "
                "grid=(%d,%d,%d) block=(%d,%d,%d) psz=%d\n",
                var, k->reg_count, k->shared_mem_size, smem, total_smem,
                grid[0], grid[1], grid[2], block[0], block[1], block[2], psz);

        auto cbuf = gpu->prepare_cbuf0(params, psz, k->cbuf0_size, k->param_base,
                                       block[0], block[1], block[2]);

        gpu->launch_kernel(cubin_base + k->code_offset, k->code_size,
                           k->reg_count, total_smem,
                           grid[0], grid[1], grid[2],
                           block[0], block[1], block[2],
                           cbuf.gpu_addr, k->cbuf0_size);
    }
};
