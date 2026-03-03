// test_nosem.cpp — Test kernel execution without semaphore sync
// Just launch, sleep, and check output
#include "gpu.h"
#include "cubin_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <unistd.h>

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

// Minimal launch — no semaphore at all
void raw_launch(GPU& gpu, uint64_t kernel_addr, uint32_t code_size,
                uint32_t reg_count, uint64_t cbuf0_gpu, uint32_t cbuf0_size,
                uint32_t grid_x) {
    QMD qmd = {};
    memset(&qmd, 0, sizeof(qmd));
    qmd.dw[4] = 0x013f0000;
    qmd.dw[9] = 0x00000000;  // no release
    qmd.dw[10] = 0x00190000;
    qmd.dw[12] = 0x02070100;
    qmd.dw[14] = 0x2f5003a4;
    qmd.dw[19] = 0x80610000;
    qmd.dw[20] = 0x00000008;
    qmd.dw[22] = 0x04000000;
    qmd.dw[58] = 0x00000011;

    uint64_t prog_s4 = kernel_addr >> 4;
    qmd.dw[32] = (uint32_t)(prog_s4 & 0xFFFFFFFF);
    qmd.dw[33] = (uint32_t)(prog_s4 >> 32) & 0x1FFFFF;
    qmd.dw[34] = (1 << 16) | 256;
    uint32_t regs = (reg_count + 1) & ~1u;
    if (regs < 16) regs = 16;
    qmd.dw[35] = 1 | (regs << 8) | (1 << 17);
    qmd.dw[36] = 8 | (3 << 11) | (0x1A << 17) | (5 << 23);
    qmd.dw[39] = grid_x;
    qmd.dw[40] = 1;
    qmd.dw[41] = 1;

    uint64_t cb_addr_s6 = cbuf0_gpu >> 6;
    uint32_t cb_size_s4 = (cbuf0_size + 15) / 16;
    qmd.dw[42] = (uint32_t)(cb_addr_s6 & 0xFFFFFFFF);
    qmd.dw[43] = ((uint32_t)(cb_addr_s6 >> 32) & 0x7FFFF) | (cb_size_s4 << 19);
    qmd.dw[59] = (uint32_t)(kernel_addr >> 8);
    qmd.dw[60] = (uint32_t)(kernel_addr >> 40);

    GPU::GpuAlloc qmd_mem = gpu.gpu_malloc(4096);
    memcpy(qmd_mem.cpu_ptr, &qmd, sizeof(qmd));
    __sync_synchronize();

    gpu.cmd_begin();
    fprintf(stderr, "  cmdq_base=%u cmdq_offset=%u compute_put=%u\n",
            gpu.cmdq_base, gpu.cmdq_offset, gpu.compute_put);
    // Always re-issue SET_OBJECT (compute engine might need reset)
    gpu.nvm(1, 0x0000, 1); gpu.nvm_data(0xCEC0);
    gpu.nvm(4, 0x0000, 1); gpu.nvm_data(0xCAB5);
    gpu.channel_setup_done = true;
    uint64_t local_window = 0x729300000000ULL;
    gpu.nvm(1, 0x07B0, 2);
    gpu.nvm_data((uint32_t)(local_window >> 32));
    gpu.nvm_data((uint32_t)(local_window & 0xFFFFFFFF));
    uint64_t shared_window = 0x729400000000ULL;
    gpu.nvm(1, 0x02A0, 2);
    gpu.nvm_data((uint32_t)(shared_window >> 32));
    gpu.nvm_data((uint32_t)(shared_window & 0xFFFFFFFF));
    gpu.nvm(1, 0x02B4, 1); gpu.nvm_data((uint32_t)(qmd_mem.gpu_addr >> 8));
    gpu.nvm(1, 0x02C0, 1); gpu.nvm_data(9);
    gpu.submit_compute();

    // Dump GPFIFO ring entries and control state
    volatile uint64_t* ring = (volatile uint64_t*)gpu.gpfifo_cpu;
    volatile uint32_t* userd = (volatile uint32_t*)((uint8_t*)gpu.gpfifo_cpu + 0x80000);
    fprintf(stderr, "  GPFIFO ring: ");
    for (uint32_t i = 0; i < gpu.compute_put; i++) {
        uint64_t e = ring[i];
        uint64_t addr = (e & 0xFFFFFFFFFC) << 0;  // bits 2-31 + bits 32-39
        uint32_t len = (e >> 42) & 0x1FFFFF;
        fprintf(stderr, "[%d]=addr=%lx len=%u ", i, (unsigned long)(e & 0xFFFFFFFFFFULL), len);
    }
    fprintf(stderr, "\n  userd GP_PUT=0x%x\n", userd[0x8C/4]);

    // Also dump the actual pushbuffer content at the new base
    uint32_t* pb = (uint32_t*)((uint8_t*)gpu.cmdq_cpu + gpu.cmdq_base);
    fprintf(stderr, "  PB@%u: ", gpu.cmdq_base);
    uint32_t ndw = (gpu.cmdq_offset - gpu.cmdq_base) / 4;
    for (uint32_t i = 0; i < ndw && i < 20; i++)
        fprintf(stderr, "%08x ", pb[i]);
    fprintf(stderr, "\n");
}

int main() {
    fprintf(stderr, "=== test_nosem: checking kernel execution without semaphore ===\n\n");

    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) return 1;
    auto* kinfo = cubin.find_kernel("add_relu_kernel");
    if (!kinfo) return 1;

    GPU gpu;
    if (!gpu.init()) return 1;
    uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());
    uint64_t kernel_addr = cubin_base + kinfo->code_offset;

    const int N = 256;

    // ==== Launch 1 ====
    fprintf(stderr, "--- Launch 1 ---\n");
    auto buf_a1 = gpu.gpu_malloc(N*2);
    auto buf_b1 = gpu.gpu_malloc(N*2);
    auto buf_y1 = gpu.gpu_malloc(N*2);
    uint16_t *a1=(uint16_t*)buf_a1.cpu_ptr, *b1=(uint16_t*)buf_b1.cpu_ptr, *y1=(uint16_t*)buf_y1.cpu_ptr;
    for (int i = 0; i < N; i++) y1[i] = 0xBEEF;
    srand(42);
    for (int i = 0; i < N; i++) {
        a1[i] = fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
        b1[i] = fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
    }
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t a,b,y; int32_t n; } args1;
    args1.a = buf_a1.gpu_addr; args1.b = buf_b1.gpu_addr;
    args1.y = buf_y1.gpu_addr; args1.n = N;
    auto cbuf1 = gpu.prepare_cbuf0(&args1, sizeof(args1), kinfo->cbuf0_size, kinfo->param_base, 256,1,1);

    raw_launch(gpu, kernel_addr, kinfo->code_size, kinfo->reg_count,
               cbuf1.gpu_addr, kinfo->cbuf0_size, 1);

    fprintf(stderr, "  Launched. Sleeping 500ms...\n");
    usleep(500000);
    __sync_synchronize();

    int beef1=0, ok1=0;
    for (int i = 0; i < N; i++) {
        if (y1[i] == 0xBEEF) beef1++;
        float exp = fmaxf(fp16_to_fp32(a1[i])+fp16_to_fp32(b1[i]), 0.0f);
        if (abs((int)y1[i] - (int)fp32_to_fp16(exp)) <= 1) ok1++;
    }
    fprintf(stderr, "  Launch 1 result: beef=%d correct=%d/%d (y[0]=0x%04x y[255]=0x%04x)\n",
            beef1, ok1, N, y1[0], y1[N-1]);

    // ==== Launch 2 (same kernel, different buffers) ====
    fprintf(stderr, "\n--- Launch 2 ---\n");
    auto buf_a2 = gpu.gpu_malloc(N*2);
    auto buf_b2 = gpu.gpu_malloc(N*2);
    auto buf_y2 = gpu.gpu_malloc(N*2);
    uint16_t *a2=(uint16_t*)buf_a2.cpu_ptr, *b2=(uint16_t*)buf_b2.cpu_ptr, *y2=(uint16_t*)buf_y2.cpu_ptr;
    for (int i = 0; i < N; i++) y2[i] = 0xBEEF;
    srand(99);
    for (int i = 0; i < N; i++) {
        a2[i] = fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
        b2[i] = fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
    }
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t a,b,y; int32_t n; } args2;
    args2.a = buf_a2.gpu_addr; args2.b = buf_b2.gpu_addr;
    args2.y = buf_y2.gpu_addr; args2.n = N;
    auto cbuf2 = gpu.prepare_cbuf0(&args2, sizeof(args2), kinfo->cbuf0_size, kinfo->param_base, 256,1,1);

    raw_launch(gpu, kernel_addr, kinfo->code_size, kinfo->reg_count,
               cbuf2.gpu_addr, kinfo->cbuf0_size, 1);

    fprintf(stderr, "  Launched. Sleeping 500ms...\n");
    usleep(500000);
    __sync_synchronize();

    int beef2=0, ok2=0;
    for (int i = 0; i < N; i++) {
        if (y2[i] == 0xBEEF) beef2++;
        float exp = fmaxf(fp16_to_fp32(a2[i])+fp16_to_fp32(b2[i]), 0.0f);
        if (abs((int)y2[i] - (int)fp32_to_fp16(exp)) <= 1) ok2++;
    }
    fprintf(stderr, "  Launch 2 result: beef=%d correct=%d/%d (y[0]=0x%04x y[255]=0x%04x)\n",
            beef2, ok2, N, y2[0], y2[N-1]);

    // ==== Launch 3 (reusing same buffers from launch 2, just re-init) ====
    fprintf(stderr, "\n--- Launch 3 (reuse buffers) ---\n");
    for (int i = 0; i < N; i++) y2[i] = 0xBEEF;
    srand(123);
    for (int i = 0; i < N; i++) {
        a2[i] = fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
        b2[i] = fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
    }
    __sync_synchronize();

    // Same args2 struct, same cbuf — just update args
    args2.a = buf_a2.gpu_addr; args2.b = buf_b2.gpu_addr;
    args2.y = buf_y2.gpu_addr; args2.n = N;
    // Rewrite cbuf0 params
    memcpy((uint8_t*)cbuf2.cpu_ptr + kinfo->param_base, &args2, sizeof(args2));
    __sync_synchronize();

    raw_launch(gpu, kernel_addr, kinfo->code_size, kinfo->reg_count,
               cbuf2.gpu_addr, kinfo->cbuf0_size, 1);

    fprintf(stderr, "  Launched. Sleeping 500ms...\n");
    usleep(500000);
    __sync_synchronize();

    int beef3=0, ok3=0;
    for (int i = 0; i < N; i++) {
        if (y2[i] == 0xBEEF) beef3++;
        float exp = fmaxf(fp16_to_fp32(a2[i])+fp16_to_fp32(b2[i]), 0.0f);
        if (abs((int)y2[i] - (int)fp32_to_fp16(exp)) <= 1) ok3++;
    }
    fprintf(stderr, "  Launch 3 result: beef=%d correct=%d/%d (y[0]=0x%04x y[255]=0x%04x)\n",
            beef3, ok3, N, y2[0], y2[N-1]);

    fprintf(stderr, "\nSummary: L1=%d/%d L2=%d/%d L3=%d/%d\n", ok1, N, ok2, N, ok3, N);
    return 0;
}
