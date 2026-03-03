// test_singlepb2.cpp — Sequential launches, each dispatch+semaphore in ONE PB entry
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

bool launch_and_verify(GPU& gpu, CubinKernel* kinfo, uint64_t kernel_addr, int iter) {
    const int N = 256;
    auto ba = gpu.gpu_malloc(N*2), bb = gpu.gpu_malloc(N*2), by = gpu.gpu_malloc(N*2);
    uint16_t *a=(uint16_t*)ba.cpu_ptr, *b=(uint16_t*)bb.cpu_ptr, *y=(uint16_t*)by.cpu_ptr;
    for (int i=0;i<N;i++) y[i]=0xBEEF;
    srand(42+iter);
    for (int i=0;i<N;i++) {
        a[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
        b[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
    }
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t a,b,y; int32_t n; } args;
    args = {ba.gpu_addr, bb.gpu_addr, by.gpu_addr, N};
    auto cbuf = gpu.prepare_cbuf0(&args, sizeof(args), kinfo->cbuf0_size, kinfo->param_base, 256,1,1);

    // Build QMD
    QMD qmd = {};
    memset(&qmd, 0, sizeof(qmd));
    qmd.dw[4] = 0x013f0000;
    qmd.dw[9] = 0x00000000;
    qmd.dw[10] = 0x00190000;
    qmd.dw[12] = 0x02070100;
    qmd.dw[14] = 0x2f5003a4;
    qmd.dw[19] = 0x80610000;
    qmd.dw[20] = 0x00000008;
    qmd.dw[22] = 0x04000000;
    qmd.dw[58] = 0x00000011;
    uint64_t ps4 = kernel_addr >> 4;
    qmd.dw[32] = (uint32_t)(ps4); qmd.dw[33] = (uint32_t)(ps4 >> 32) & 0x1FFFFF;
    uint32_t regs = (kinfo->reg_count + 1) & ~1u; if (regs < 16) regs = 16;
    qmd.dw[34] = (1<<16)|256; qmd.dw[35] = 1|(regs<<8)|(1<<17);
    qmd.dw[36] = 8|(3<<11)|(0x1A<<17)|(5<<23);
    qmd.dw[39] = 1; qmd.dw[40] = 1; qmd.dw[41] = 1;
    uint64_t cbs6 = cbuf.gpu_addr >> 6;
    uint32_t cbss4 = (kinfo->cbuf0_size + 15) / 16;
    qmd.dw[42] = (uint32_t)(cbs6); qmd.dw[43] = ((uint32_t)(cbs6>>32)&0x7FFFF)|(cbss4<<19);
    qmd.dw[59] = (uint32_t)(kernel_addr >> 8); qmd.dw[60] = (uint32_t)(kernel_addr >> 40);

    auto qmem = gpu.gpu_malloc(4096);
    memcpy(qmem.cpu_ptr, &qmd, sizeof(qmd));
    __sync_synchronize();

    // Build single pushbuffer: preamble + SEND_PCAS + SEM_RELEASE
    gpu.cmd_begin();

    if (!gpu.channel_setup_done) {
        gpu.nvm(1, 0x0000, 1); gpu.nvm_data(0xCEC0);
        gpu.nvm(4, 0x0000, 1); gpu.nvm_data(0xCAB5);
        gpu.channel_setup_done = true;
    }
    uint64_t lw = 0x729300000000ULL, sw = 0x729400000000ULL;
    gpu.nvm(1, 0x07B0, 2); gpu.nvm_data((uint32_t)(lw>>32)); gpu.nvm_data((uint32_t)(lw));
    gpu.nvm(1, 0x02A0, 2); gpu.nvm_data((uint32_t)(sw>>32)); gpu.nvm_data((uint32_t)(sw));
    gpu.nvm(1, 0x02B4, 1); gpu.nvm_data((uint32_t)(qmem.gpu_addr >> 8));
    gpu.nvm(1, 0x02C0, 1); gpu.nvm_data(9);

    // HOST semaphore IN SAME pushbuffer
    gpu.init_semaphore();
    gpu.sem_counter++;
    gpu.nvm(0, 0x005C, 5);
    gpu.nvm_data((uint32_t)(gpu.sem_alloc.gpu_addr & 0xFFFFFFFF));
    gpu.nvm_data((uint32_t)(gpu.sem_alloc.gpu_addr >> 32));
    gpu.nvm_data((uint32_t)(gpu.sem_counter & 0xFFFFFFFF));
    gpu.nvm_data((uint32_t)(gpu.sem_counter >> 32));
    gpu.nvm_data(0x01000001);

    gpu.submit_compute();

    // Wait for semaphore
    volatile uint64_t* sem64 = (volatile uint64_t*)gpu.sem_alloc.cpu_ptr;
    int spins = 0;
    while (*sem64 < gpu.sem_counter) {
        __sync_synchronize();
        if (++spins > 100000000) break;
    }
    __sync_synchronize();

    // Small delay to ensure kernel completes (HOST semaphore fires before kernel finishes)
    usleep(100);

    __sync_synchronize();

    // Verify
    int mis = 0;
    for (int i=0;i<N;i++) {
        float exp = fmaxf(fp16_to_fp32(a[i])+fp16_to_fp32(b[i]), 0.0f);
        int diff = abs((int)y[i]-(int)fp32_to_fp16(exp));
        if (diff > 1) {
            if (mis < 3) fprintf(stderr, "    MIS[%d]: y=0x%04x(%.4f) exp=0x%04x(%.4f) diff=%d\n",
                                  i, y[i], fp16_to_fp32(y[i]), fp32_to_fp16(exp), exp, diff);
            mis++;
        }
    }
    fprintf(stderr, "  Launch %d: %s (spins=%d sem=0x%lx mis=%d)\n",
            iter, mis==0?"PASS":"FAIL", spins, (unsigned long)*sem64, mis);
    return mis == 0;
}

int main() {
    fprintf(stderr, "=== test_singlepb2: sequential single-PB launches ===\n\n");

    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) return 1;
    auto* kinfo = cubin.find_kernel("add_relu_kernel");
    if (!kinfo) return 1;

    GPU gpu;
    if (!gpu.init()) return 1;
    uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());
    uint64_t kernel_addr = cubin_base + kinfo->code_offset;

    bool ok = true;
    for (int i = 0; i < 5; i++) {
        if (!launch_and_verify(gpu, kinfo, kernel_addr, i)) {
            ok = false; break;
        }
    }
    fprintf(stderr, "\n%s\n", ok ? "ALL 5 PASS" : "FAILED");
    return ok ? 0 : 1;
}
