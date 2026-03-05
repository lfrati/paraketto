// test_cutlass_cuda.cu — Launch CUTLASS cubin via CUDA driver API
// to verify params are correct independently of cudaless launch path.
//
// If PASS: params correct, issue is cudaless launch (QMD/cbuf0)
// If FAIL: params construction is wrong
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

static uint16_t fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
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
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        exp++; frac &= 0x3FF;
    } else if (exp == 31) {
        uint32_t v = sign | 0x7F800000 | (frac << 13);
        float f; memcpy(&f, &v, 4); return f;
    }
    uint32_t v = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float f; memcpy(&f, &v, 4); return f;
}

static void fill_random_fp16(uint16_t* data, int count, unsigned seed) {
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245 + 12345;
        float val = ((float)(seed >> 16) / 65536.0f - 0.5f) * 2.0f;
        data[i] = fp32_to_fp16(val);
    }
}

// Find kernel name from cubin ELF by scanning for ".text._Z" sections
static std::string find_nn_kernel_name(const char* cubin_path) {
    FILE* f = fopen(cubin_path, "rb");
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
        // NN_64x64_64: has "64ELi64ELi64E", no "GemmBatched", no "Congruous"
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

    // Find kernel name
    std::string kname = find_nn_kernel_name("cutlass_gemm.cubin");
    if (kname.empty()) { fprintf(stderr, "Can't find NN kernel in cubin\n"); return 1; }
    fprintf(stderr, "Kernel: %s\n", kname.c_str());

    CUmodule mod;
    CUresult res = cuModuleLoad(&mod, "cutlass_gemm.cubin");
    if (res != CUDA_SUCCESS) { fprintf(stderr, "cuModuleLoad: %d\n", res); return 1; }

    CUfunction func;
    res = cuModuleGetFunction(&func, mod, kname.c_str());
    if (res != CUDA_SUCCESS) { fprintf(stderr, "cuModuleGetFunction: %d\n", res); return 1; }

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * 2);
    cudaMalloc(&d_B, K * N * 2);
    cudaMalloc(&d_C, M * N * 2);

    uint16_t h_A[M * K], h_B[K * N];
    fill_random_fp16(h_A, M * K, 42);
    fill_random_fp16(h_B, K * N, 123);
    cudaMemcpy(d_A, h_A, M * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * 2);

    // Build params: row→col swap: CUTLASS M=n, N=m, A=B, B=A
    uint8_t params[512];
    int grid[3], block[3], smem;
    int psz = cutlass_params_nn_64x64_64_s6(
        params, 512, grid, block, &smem,
        N, M, K, (uint64_t)d_B, N, (uint64_t)d_A, K, (uint64_t)d_C, N,
        0x3C00, 0x0000);

    fprintf(stderr, "psz=%d grid=(%d,%d,%d) block=(%d,%d,%d) smem=%d\n",
            psz, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem);

    cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);

    // Launch with CU_LAUNCH_PARAM_BUFFER
    size_t psz_t = (size_t)psz;
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, params,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &psz_t,
        CU_LAUNCH_PARAM_END
    };

    res = cuLaunchKernel(func, grid[0], grid[1], grid[2],
                         block[0], block[1], block[2], smem, 0, nullptr, extra);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "cuLaunchKernel: %d\n", res); return 1; }
    cudaDeviceSynchronize();
    fprintf(stderr, "Kernel completed.\n");

    uint16_t h_C[M * N];
    cudaMemcpy(h_C, d_C, M * N * 2, cudaMemcpyDeviceToHost);

    // CPU reference
    uint16_t cpu_C[M * N] = {};
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int p = 0; p < K; p++)
                sum += fp16_to_fp32(h_A[i * K + p]) * fp16_to_fp32(h_B[p * N + j]);
            cpu_C[i * N + j] = fp32_to_fp16(sum);
        }

    int errors = 0;
    float max_rel = 0;
    for (int i = 0; i < M * N; i++) {
        float g = fp16_to_fp32(h_C[i]);
        float c = fp16_to_fp32(cpu_C[i]);
        float diff = fabsf(g - c);
        float rel = (fabsf(c) > 1e-6f) ? diff / fabsf(c) : diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > 0.01f && rel > 0.02f) {
            if (errors < 5)
                printf("  MISMATCH [%d]: gpu=%.6f cpu=%.6f\n", i, g, c);
            errors++;
        }
    }

    printf("\nFirst 8: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", fp16_to_fp32(h_C[i]));
    printf("\nCPU   8: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", fp16_to_fp32(cpu_C[i]));
    printf("\n");

    if (errors == 0)
        printf("\nCUDA DRIVER API: PASS (max_rel=%.4f) — params OK, issue is cudaless launch\n", max_rel);
    else
        printf("\nCUDA DRIVER API: FAIL (%d/%d errors, max_rel=%.4f) — PARAMS ARE WRONG\n",
               errors, M * N, max_rel);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cuModuleUnload(mod);
    return errors > 0 ? 1 : 0;
}
