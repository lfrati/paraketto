// test_cutlass_ptx.cu — Load CUTLASS via PTX (JIT-compiled) vs cubin
// If PTX PASS + cubin FAIL → nvcc SM120 SASS codegen bug
// If both FAIL → params or launch issue
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

// Find NN_64x64_64 kernel name from cubin ELF
static std::string find_nn_kernel_name_elf(const char* path) {
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

// Find NN_64x64_64 kernel name from PTX
static std::string find_nn_kernel_name_ptx(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return "";
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
        fclose(f);
        return name;
    }
    fclose(f);
    return "";
}

static int test_kernel(CUfunction func, const char* label,
                       half* d_A, half* d_B, half* d_C,
                       const uint16_t* h_A, const uint16_t* h_B,
                       int M, int N, int K) {
    cudaMemcpy(d_A, h_A, M * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * 2);

    uint8_t params[512];
    int grid[3], block[3], smem;
    int psz = cutlass_params_nn_64x64_64_s6(
        params, 512, grid, block, &smem,
        N, M, K, (uint64_t)d_B, N, (uint64_t)d_A, K, (uint64_t)d_C, N,
        0x3C00, 0x0000);

    cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);

    size_t psz_t = (size_t)psz;
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, params,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &psz_t,
        CU_LAUNCH_PARAM_END
    };
    CUresult res = cuLaunchKernel(func, grid[0], grid[1], grid[2],
                                   block[0], block[1], block[2], smem, 0, nullptr, extra);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "%s: cuLaunchKernel: %d\n", label, res); return -1; }
    cudaDeviceSynchronize();

    uint16_t h_C[M * N];
    cudaMemcpy(h_C, d_C, M * N * 2, cudaMemcpyDeviceToHost);

    // CPU reference
    int errors = 0;
    float max_rel = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int p = 0; p < K; p++)
                sum += fp16_to_fp32(h_A[i * K + p]) * fp16_to_fp32(h_B[p * N + j]);
            float g = fp16_to_fp32(h_C[i * N + j]);
            float diff = fabsf(g - sum);
            float rel = (fabsf(sum) > 1e-6f) ? diff / fabsf(sum) : diff;
            if (rel > max_rel) max_rel = rel;
            if (diff > 0.01f && rel > 0.02f) errors++;
        }

    if (errors == 0)
        printf("%s: PASS (%d elements, max_rel=%.4f)\n", label, M * N, max_rel);
    else
        printf("%s: FAIL (%d/%d errors, max_rel=%.4f)\n", label, errors, M * N, max_rel);
    return errors;
}

int main() {
    const int M = 64, N = 64, K = 64;

    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    cuCtxCreate(&ctx, &ctxParams, 0, dev);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * 2);
    cudaMalloc(&d_B, K * N * 2);
    cudaMalloc(&d_C, M * N * 2);

    uint16_t h_A[M * K], h_B[K * N];
    fill_random_fp16(h_A, M * K, 42);
    fill_random_fp16(h_B, K * N, 123);

    // Test 1: cubin (SM120 offline SASS)
    {
        std::string kname = find_nn_kernel_name_elf("cutlass_gemm.cubin");
        if (kname.empty()) { printf("cubin: Can't find kernel\n"); }
        else {
            CUmodule mod;
            CUresult res = cuModuleLoad(&mod, "cutlass_gemm.cubin");
            if (res != CUDA_SUCCESS) printf("cubin: cuModuleLoad failed: %d\n", res);
            else {
                CUfunction func;
                res = cuModuleGetFunction(&func, mod, kname.c_str());
                if (res != CUDA_SUCCESS) printf("cubin: cuModuleGetFunction failed: %d\n", res);
                else test_kernel(func, "cubin (SM120 SASS)", d_A, d_B, d_C, h_A, h_B, M, N, K);
                cuModuleUnload(mod);
            }
        }
    }

    // Test 2: PTX (JIT-compiled to SM120)
    {
        std::string kname = find_nn_kernel_name_ptx("cutlass_gemm.ptx");
        if (kname.empty()) { printf("ptx: Can't find kernel\n"); }
        else {
            fprintf(stderr, "PTX kernel: %s\n", kname.c_str());
            CUmodule mod;
            CUresult res = cuModuleLoad(&mod, "cutlass_gemm.ptx");
            if (res != CUDA_SUCCESS) printf("ptx: cuModuleLoad failed: %d\n", res);
            else {
                CUfunction func;
                res = cuModuleGetFunction(&func, mod, kname.c_str());
                if (res != CUDA_SUCCESS) printf("ptx: cuModuleGetFunction failed: %d\n", res);
                else test_kernel(func, "ptx  (JIT SM120)", d_A, d_B, d_C, h_A, h_B, M, N, K);
                cuModuleUnload(mod);
            }
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
