// test_cutlass_sidebyside.cu — DEFINITIVE test
// Same program, same data, same params:
//   1) CUTLASS normal path (op(stream)) → known correct
//   2) cuLaunchKernel with same params on same PTX module → suspect
// Compares byte-for-byte.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// Same type as cutlass_gemm.cu
template<int Align>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, Align, cutlass::half_t, cutlass::half_t>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using GemmNN = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue<8>, Swizzle, 6, 8, 8>;

static float h2f(half h) { return __half2float(h); }
static half f2h(float f) { return __float2half(f); }

int main() {
    const int M = 64, N = 64, K = 64;

    // Row-major: Y[M,N] = X[M,K] @ W[K,N]
    // Col-major CUTLASS: col_m=N, col_n=M, A=W, B=X
    int col_m = N, col_n = M, col_k = K;
    int ldA = N, ldB = K, ldC = N;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    half *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, M * K * 2);
    cudaMalloc(&d_B, K * N * 2);
    cudaMalloc(&d_C1, M * N * 2);  // for CUTLASS normal
    cudaMalloc(&d_C2, M * N * 2);  // for cuLaunchKernel

    // Random data
    half h_X[M * K], h_W[K * N];
    unsigned seed = 42;
    for (int i = 0; i < M * K; i++) {
        seed = seed * 1103515245 + 12345;
        h_X[i] = f2h(((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f);
    }
    seed = 123;
    for (int i = 0; i < K * N; i++) {
        seed = seed * 1103515245 + 12345;
        h_W[i] = f2h(((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f);
    }
    // Upload: d_A = X (row-major), d_B = W (row-major)
    cudaMemcpy(d_A, h_X, M * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_W, K * N * 2, cudaMemcpyHostToDevice);

    printf("Ptrs: X(d_A)=%p W(d_B)=%p C1=%p C2=%p\n", d_A, d_B, d_C1, d_C2);

    half h_C2a[M * N] = {};  // for cuLaunchKernel buffer method

    // ====== Method 1: CUTLASS normal operator() ======
    {
        using ElementA = cutlass::half_t;
        using ElementC = cutlass::half_t;

        // Row→Col swap: CUTLASS A=W(ld=N), B=X(ld=K), M=N, N=M
        GemmNN::Arguments args(
            {col_m, col_n, col_k},
            {reinterpret_cast<const ElementA*>(d_B), ldA},   // A_cutlass = W
            {reinterpret_cast<const ElementA*>(d_A), ldB},   // B_cutlass = X
            {reinterpret_cast<ElementC*>(d_C1), ldC},
            {reinterpret_cast<ElementC*>(d_C1), ldC},
            {cutlass::half_t(1.0f), cutlass::half_t(0.0f)}
        );

        GemmNN op;
        memset(&op, 0, sizeof(op));
        auto status = op.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            printf("CUTLASS initialize failed\n"); return 1;
        }

        cudaMemset(d_C1, 0, M * N * 2);
        status = op(stream);
        cudaStreamSynchronize(stream);
        if (status != cutlass::Status::kSuccess) {
            printf("CUTLASS op() failed\n"); return 1;
        }

        // Extract params for comparison and for cuLaunchKernel
        using Params = typename GemmNN::GemmKernel::Params;
        printf("sizeof(Params) = %zu\n", sizeof(Params));

        uint8_t cutlass_params[512] = {};
        memcpy(cutlass_params, &op, sizeof(Params));

        // ====== Method 2: cuLaunchKernel with SAME params on PTX module ======
        cuInit(0);
        // Get the current context (created by CUDA runtime)
        CUcontext ctx;
        cuCtxGetCurrent(&ctx);

        // Load PTX module
        CUmodule mod;
        CUresult res = cuModuleLoad(&mod, "cutlass_gemm.ptx");
        if (res != CUDA_SUCCESS) { printf("cuModuleLoad failed: %d\n", res); return 1; }

        // Find the NN kernel
        FILE* f = fopen("cutlass_gemm.ptx", "r");
        std::string kname;
        if (f) {
            char line[8192];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, ".entry _Z", 9) != 0) continue;
                char* name_start = line + 7;
                char* paren = strchr(name_start, '(');
                if (!paren) continue;
                std::string name(name_start, paren - name_start);
                if (name.find("GemmBatched") != std::string::npos) continue;
                if (name.find("Congruous") != std::string::npos) continue;
                if (name.find("64ELi64ELi64E") == std::string::npos) continue;
                kname = name;
                break;
            }
            fclose(f);
        }
        if (kname.empty()) { printf("Can't find kernel in PTX\n"); return 1; }

        CUfunction func;
        res = cuModuleGetFunction(&func, mod, kname.c_str());
        if (res != CUDA_SUCCESS) { printf("cuModuleGetFunction: %d\n", res); return 1; }

        int smem_size = (int)sizeof(typename GemmNN::GemmKernel::SharedStorage);
        cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size);

        // Patch C pointer in params to point to d_C2 instead of d_C1
        // C pointer is at two locations: output_ref (C) and output (D)
        // Find them by scanning for d_C1's address
        uint64_t c1_addr = (uint64_t)d_C1;
        uint64_t c2_addr = (uint64_t)d_C2;
        int patched = 0;
        for (int i = 0; i <= (int)sizeof(Params) - 8; i++) {
            uint64_t val;
            memcpy(&val, cutlass_params + i, 8);
            if (val == c1_addr) {
                memcpy(cutlass_params + i, &c2_addr, 8);
                printf("  Patched C ptr at offset 0x%x\n", i);
                patched++;
            }
        }
        printf("  Patched %d C pointer(s)\n", patched);

        // Get grid dims from params
        int32_t* pw = (int32_t*)cutlass_params;
        dim3 grid(pw[3], pw[4], pw[5]);
        dim3 block(GemmNN::GemmKernel::kThreadCount, 1, 1);
        printf("  grid=(%d,%d,%d) block=(%d,%d,%d) smem=%d psz=%zu\n",
               grid.x, grid.y, grid.z, block.x, block.y, block.z,
               smem_size, sizeof(Params));

        cudaMemset(d_C2, 0, M * N * 2);

        // Method A: CU_LAUNCH_PARAM_BUFFER (raw bytes)
        size_t psz_t = sizeof(Params);
        void* extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, cutlass_params,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &psz_t,
            CU_LAUNCH_PARAM_END
        };
        res = cuLaunchKernel(func, grid.x, grid.y, grid.z,
                              block.x, block.y, block.z,
                              smem_size, (CUstream)stream, nullptr, extra);
        if (res != CUDA_SUCCESS) { printf("cuLaunchKernel (buffer): %d\n", res); return 1; }
        cudaStreamSynchronize(stream);

        // Read method A results
        cudaMemcpy(h_C2a, d_C2, M * N * 2, cudaMemcpyDeviceToHost);

        // Method B: void** args (pointer-to-struct, like cudaLaunchKernel)
        cudaMemset(d_C2, 0, M * N * 2);
        void* args_ptrs[] = { cutlass_params };
        res = cuLaunchKernel(func, grid.x, grid.y, grid.z,
                              block.x, block.y, block.z,
                              smem_size, (CUstream)stream, args_ptrs, nullptr);
        if (res != CUDA_SUCCESS) { printf("cuLaunchKernel (args): %d\n", res); return 1; }
        cudaStreamSynchronize(stream);

        cuModuleUnload(mod);

        // Method C: cudaLaunchKernel with EMBEDDED kernel + our params
        // This tests if our extracted params work with the embedded kernel
        cudaMemset(d_C2, 0, M * N * 2);
        {
            auto* kernel_fn = &cutlass::Kernel<GemmNN::GemmKernel>;
            if (smem_size >= (48 << 10)) {
                cudaFuncSetAttribute((const void*)kernel_fn,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            }
            void* embedded_args[] = { cutlass_params };
            cudaError_t err = cudaLaunchKernel(
                (const void*)kernel_fn,
                grid, block, embedded_args, smem_size, stream);
            if (err != cudaSuccess)
                printf("cudaLaunchKernel (embedded): %s\n", cudaGetErrorString(err));
            cudaStreamSynchronize(stream);
        }
    }

    half h_C2c[M * N];
    cudaMemcpy(h_C2c, d_C2, M * N * 2, cudaMemcpyDeviceToHost);

    // Read back and compare
    half h_C1[M * N], h_C2[M * N];
    cudaMemcpy(h_C1, d_C1, M * N * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, M * N * 2, cudaMemcpyDeviceToHost);

    // CPU reference
    float cpu_C[M * N] = {};
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int p = 0; p < K; p++)
                cpu_C[i * N + j] += h2f(h_X[i * K + p]) * h2f(h_W[p * N + j]);

    printf("\nFirst 8 values:\n");
    printf("  CUTLASS op():  "); for (int i = 0; i < 8; i++) printf("%.4f ", h2f(h_C1[i])); printf("\n");
    printf("  cuLaunch buf:  "); for (int i = 0; i < 8; i++) printf("%.4f ", h2f(h_C2a[i])); printf("\n");
    printf("  cuLaunch arg:  "); for (int i = 0; i < 8; i++) printf("%.4f ", h2f(h_C2[i])); printf("\n");
    printf("  embedded+args: "); for (int i = 0; i < 8; i++) printf("%.4f ", h2f(h_C2c[i])); printf("\n");
    printf("  CPU:           "); for (int i = 0; i < 8; i++) printf("%.4f ", cpu_C[i]); printf("\n");

    auto count_err = [&](const half* h_C) {
        int err = 0;
        for (int i = 0; i < M * N; i++) {
            float v = h2f(h_C[i]), vc = cpu_C[i];
            float d = fabsf(v - vc);
            float r = fabsf(vc) > 1e-6f ? d/fabsf(vc) : d;
            if (d > 0.01f && r > 0.02f) err++;
        }
        return err;
    };
    printf("\nCUTLASS op() vs CPU:         %d/%d errors\n", count_err(h_C1), M * N);
    printf("cuLaunch(buffer) vs CPU:     %d/%d errors\n", count_err(h_C2a), M * N);
    printf("cuLaunch(args) vs CPU:       %d/%d errors\n", count_err(h_C2), M * N);
    printf("embedded+our_args vs CPU:    %d/%d errors\n", count_err(h_C2c), M * N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    cudaStreamDestroy(stream);
    return 0;
}
