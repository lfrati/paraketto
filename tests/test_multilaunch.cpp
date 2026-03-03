#include <unistd.h>
// test_multilaunch.cpp — Minimal test: launch add_relu twice to diagnose hang
#include "gpu.h"
#include "cubin_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

static uint16_t fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127;
    uint32_t frac = x & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) return sign;
    return sign | ((exp + 15) << 10) | (frac >> 13);
}
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) { if (frac == 0) { float r; uint32_t v = sign; memcpy(&r, &v, 4); return r; }
        exp = 1; while (!(frac & 0x400)) { frac <<= 1; exp--; } frac &= 0x3FF;
    } else if (exp == 31) { float r; uint32_t v = sign | 0x7F800000 | (frac << 13); memcpy(&r, &v, 4); return r; }
    uint32_t v = sign | ((exp + 112) << 23) | (frac << 13);
    float r; memcpy(&r, &v, 4); return r;
}

bool run_add_relu(GPU& gpu, CubinKernel* kinfo, uint64_t cubin_base, int iteration) {
    uint64_t kernel_addr = cubin_base + kinfo->code_offset;
    const int N = 256;  // small
    const size_t buf_bytes = N * sizeof(uint16_t);

    auto buf_a = gpu.gpu_malloc(buf_bytes);
    auto buf_b = gpu.gpu_malloc(buf_bytes);
    auto buf_y = gpu.gpu_malloc(buf_bytes);

    uint16_t* a = (uint16_t*)buf_a.cpu_ptr;
    uint16_t* b = (uint16_t*)buf_b.cpu_ptr;
    uint16_t* y = (uint16_t*)buf_y.cpu_ptr;
    for (int i = 0; i < N; i++) y[i] = 0xBEEF;

    srand(42 + iteration);
    for (int i = 0; i < N; i++) {
        a[i] = fp32_to_fp16(((float)rand() / RAND_MAX) * 4.0f - 2.0f);
        b[i] = fp32_to_fp16(((float)rand() / RAND_MAX) * 4.0f - 2.0f);
    }
    __sync_synchronize();

    struct __attribute__((packed)) {
        uint64_t a_ptr, b_ptr, y_ptr;
        int32_t n;
    } args;
    args.a_ptr = buf_a.gpu_addr;
    args.b_ptr = buf_b.gpu_addr;
    args.y_ptr = buf_y.gpu_addr;
    args.n = N;

    auto cbuf = gpu.prepare_cbuf0(&args, sizeof(args), kinfo->cbuf0_size, kinfo->param_base, 256, 1, 1);

    fprintf(stderr, "\n--- Launch %d: grid=(%d,1,1) block=(256,1,1) ---\n", iteration, (N+255)/256);
    fprintf(stderr, "  sem_counter before: %lu\n", (unsigned long)gpu.sem_counter);
    fprintf(stderr, "  cmdq_offset before cmd_begin: %u\n", gpu.cmdq_offset);

    gpu.launch_kernel(kernel_addr, kinfo->code_size,
                      kinfo->reg_count, kinfo->shared_mem_size,
                      (N+255)/256, 1, 1, 256, 1, 1,
                      cbuf.gpu_addr, kinfo->cbuf0_size);

    fprintf(stderr, "  sem_counter after launch: %lu\n", (unsigned long)gpu.sem_counter);
    fprintf(stderr, "  compute_put after launch: %u\n", gpu.compute_put);

    gpu.wait_kernel();
    usleep(100000);  // 100ms delay to let kernel actually complete

    // Verify
    __sync_synchronize();
    int mis = 0;
    for (int i = 0; i < N; i++) {
        float expected = fmaxf(fp16_to_fp32(a[i]) + fp16_to_fp32(b[i]), 0.0f);
        uint16_t exp_h = fp32_to_fp16(expected);
        if (abs((int)y[i] - (int)exp_h) > 1) mis++;
    }
    fprintf(stderr, "  Launch %d: %s (%d mismatches)\n", iteration, mis == 0 ? "PASS" : "FAIL", mis);
    return mis == 0;
}

int main() {
    fprintf(stderr, "=== test_multilaunch: same kernel launched 3 times ===\n\n");

    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) return 1;
    auto* kinfo = cubin.find_kernel("add_relu_kernel");
    if (!kinfo) return 1;

    GPU gpu;
    if (!gpu.init()) return 1;
    uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());

    bool ok = true;
    for (int i = 0; i < 3; i++) {
        if (!run_add_relu(gpu, kinfo, cubin_base, i)) {
            ok = false;
            break;
        }
    }
    fprintf(stderr, "\n%s\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
