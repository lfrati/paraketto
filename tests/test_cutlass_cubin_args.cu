// test_cutlass_cubin_args.cu — Test cubin with void** args (not CU_LAUNCH_PARAM_BUFFER)
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

extern "C" int cutlass_params_nn_64x64_64_s6(
    void* out, int out_sz, int* grid, int* block, int* smem,
    int M, int N, int K, uint64_t A, int ldA, uint64_t B, int ldB,
    uint64_t C, int ldC, uint16_t alpha, uint16_t beta);

static float h2f(half h) { return __half2float(h); }
static half f2h(float f) { return __float2half(f); }

static std::string find_nn_kernel_name(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> data(sz);
    if (fread(data.data(), 1, sz, f) != (size_t)sz) { fclose(f); return ""; }
    fclose(f);
    for (long i = 0; i < sz - 20; i++) {
        if (memcmp(data.data() + i, ".text._Z", 8) != 0) continue;
        const char* start = data.data() + i + 6;
        const char* end = start;
        while (end < data.data() + sz && *end) end++;
        std::string name(start, end - start);
        if (name.find("GemmBatched") != std::string::npos) continue;
        if (name.find("Congruous") != std::string::npos) continue;
        if (name.find("64ELi64ELi64E") == std::string::npos) continue;
        return name;
    }
    return "";
}

int main() {
    const int M = 64, N = 64, K = 64;

    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    cuCtxCreate(&ctx, &ctxParams, 0, dev);

    std::string kname = find_nn_kernel_name("cutlass_gemm.cubin");
    if (kname.empty()) { printf("Can't find kernel\n"); return 1; }

    CUmodule mod;
    if (cuModuleLoad(&mod, "cutlass_gemm.cubin") != CUDA_SUCCESS) { printf("cuModuleLoad failed\n"); return 1; }
    CUfunction func;
    if (cuModuleGetFunction(&func, mod, kname.c_str()) != CUDA_SUCCESS) { printf("cuModuleGetFunction failed\n"); return 1; }

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * 2);
    cudaMalloc(&d_B, K * N * 2);
    cudaMalloc(&d_C, M * N * 2);

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
    cudaMemcpy(d_A, h_X, M * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_W, K * N * 2, cudaMemcpyHostToDevice);

    uint8_t params[512] __attribute__((aligned(16)));
    int grid[3], block[3], smem;
    int psz = cutlass_params_nn_64x64_64_s6(
        params, 512, grid, block, &smem,
        N, M, K, (uint64_t)d_B, N, (uint64_t)d_A, K, (uint64_t)d_C, N,
        0x3C00, 0x0000);

    cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);

    // Test A: CU_LAUNCH_PARAM_BUFFER (known broken)
    cudaMemset(d_C, 0, M * N * 2);
    {
        size_t psz_t = (size_t)psz;
        void* extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, params,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &psz_t,
            CU_LAUNCH_PARAM_END
        };
        cuLaunchKernel(func, grid[0], grid[1], grid[2],
                       block[0], block[1], block[2], smem, 0, nullptr, extra);
        cudaDeviceSynchronize();
    }
    half h_Ca[M * N];
    cudaMemcpy(h_Ca, d_C, M * N * 2, cudaMemcpyDeviceToHost);

    // Test B: void** args
    cudaMemset(d_C, 0, M * N * 2);
    {
        void* args[] = { params };
        cuLaunchKernel(func, grid[0], grid[1], grid[2],
                       block[0], block[1], block[2], smem, 0, args, nullptr);
        cudaDeviceSynchronize();
    }
    half h_Cb[M * N];
    cudaMemcpy(h_Cb, d_C, M * N * 2, cudaMemcpyDeviceToHost);

    // CPU reference
    float cpu_C[M * N] = {};
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int p = 0; p < K; p++)
                cpu_C[i * N + j] += h2f(h_X[i * K + p]) * h2f(h_W[p * N + j]);

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

    printf("CUBIN + CU_LAUNCH_PARAM_BUFFER: %d/%d errors\n", count_err(h_Ca), M*N);
    printf("CUBIN + void** args:            %d/%d errors\n", count_err(h_Cb), M*N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cuModuleUnload(mod);
    return 0;
}
