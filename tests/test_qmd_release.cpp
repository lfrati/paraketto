// test_qmd_release.cpp — Test QMD release semaphore configurations
// Launches add_relu once, using ONLY QMD release (no HOST release)
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

// Custom launch that tests QMD release without HOST release
bool launch_qmd_only(GPU& gpu, uint64_t kernel_addr, uint32_t code_size,
                     uint32_t reg_count, uint32_t shared_mem,
                     uint32_t grid_x, uint64_t cbuf0_gpu_addr, uint32_t cbuf0_size,
                     uint32_t dw9_value, const char* config_name) {
    gpu.init_semaphore();

    QMD qmd = {};
    memset(&qmd, 0, sizeof(qmd));

    qmd.dw[4] = 0x013f0000;
    qmd.dw[9] = dw9_value;  // TEST VARIABLE
    qmd.dw[10] = 0x00190000;
    qmd.dw[12] = 0x02070100;
    qmd.dw[14] = 0x2f5003a4;
    qmd.dw[19] = 0x80610000;
    qmd.dw[20] = 0x00000008;
    qmd.dw[22] = 0x04000000;
    qmd.dw[58] = 0x00000011;

    gpu.sem_counter++;

    // Set QMD release semaphore
    qmd.dw[15] = (uint32_t)(gpu.sem_alloc.gpu_addr & 0xFFFFFFFF);
    qmd.dw[16] = (uint32_t)(gpu.sem_alloc.gpu_addr >> 32);
    qmd.dw[17] = (uint32_t)(gpu.sem_counter & 0xFFFFFFFF);
    qmd.dw[18] = (uint32_t)(gpu.sem_counter >> 32);

    // Program address
    uint64_t prog_s4 = kernel_addr >> 4;
    qmd.dw[32] = (uint32_t)(prog_s4 & 0xFFFFFFFF);
    qmd.dw[33] = (uint32_t)(prog_s4 >> 32) & 0x1FFFFF;

    qmd.dw[34] = (1 << 16) | 256;  // block 256x1
    qmd.dw[35] = 1 | (16 << 8) | (1 << 17);  // z=1, regs=16, barrier=1

    qmd.dw[36] = 8 | (3 << 11) | (0x1A << 17) | (5 << 23);  // smem

    qmd.dw[39] = grid_x;
    qmd.dw[40] = 1;
    qmd.dw[41] = 1;

    uint64_t cb_addr_s6 = cbuf0_gpu_addr >> 6;
    uint32_t cb_size_s4 = (cbuf0_size + 15) / 16;
    qmd.dw[42] = (uint32_t)(cb_addr_s6 & 0xFFFFFFFF);
    qmd.dw[43] = ((uint32_t)(cb_addr_s6 >> 32) & 0x7FFFF) | (cb_size_s4 << 19);

    qmd.dw[59] = (uint32_t)(kernel_addr >> 8);
    qmd.dw[60] = (uint32_t)(kernel_addr >> 40);

    fprintf(stderr, "\n--- Config: %s (DW[9]=0x%08x) ---\n", config_name, dw9_value);
    fprintf(stderr, "  sem_addr=0x%lx payload=0x%lx\n",
            (unsigned long)gpu.sem_alloc.gpu_addr, (unsigned long)gpu.sem_counter);

    // Dump key QMD DWs
    fprintf(stderr, "  DW[9]=0x%08x DW[15]=0x%08x DW[16]=0x%08x DW[17]=0x%08x DW[18]=0x%08x\n",
            qmd.dw[9], qmd.dw[15], qmd.dw[16], qmd.dw[17], qmd.dw[18]);

    // Allocate QMD
    GPU::GpuAlloc qmd_mem = gpu.gpu_malloc(4096);
    memcpy(qmd_mem.cpu_ptr, &qmd, sizeof(qmd));
    __sync_synchronize();

    // Build command buffer
    gpu.cmd_begin();

    if (!gpu.channel_setup_done) {
        gpu.nvm(1, 0x0000, 1); gpu.nvm_data(0xCEC0);
        gpu.nvm(4, 0x0000, 1); gpu.nvm_data(0xCAB5);
        gpu.channel_setup_done = true;
    }

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

    // NO HOST semaphore release — relying solely on QMD release

    // Poll for 3 seconds
    volatile uint64_t* sem64 = (volatile uint64_t*)gpu.sem_alloc.cpu_ptr;
    volatile uint32_t* sem32 = (volatile uint32_t*)gpu.sem_alloc.cpu_ptr;
    for (int attempt = 0; attempt < 30; attempt++) {
        __sync_synchronize();
        uint64_t val = *sem64;
        fprintf(stderr, "  poll[%d]: sem64=0x%lx sem32[0..3]=%08x %08x %08x %08x\n",
                attempt, (unsigned long)val, sem32[0], sem32[1], sem32[2], sem32[3]);
        if (val >= gpu.sem_counter) {
            fprintf(stderr, "  QMD release FIRED! after %d polls\n", attempt);
            return true;
        }
        usleep(100000);  // 100ms
    }
    fprintf(stderr, "  QMD release NEVER fired after 3 seconds\n");
    return false;
}

int main() {
    fprintf(stderr, "=== test_qmd_release: testing QMD release semaphore configs ===\n\n");

    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) return 1;
    auto* kinfo = cubin.find_kernel("add_relu_kernel");
    if (!kinfo) return 1;

    GPU gpu;
    if (!gpu.init()) return 1;
    uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());
    uint64_t kernel_addr = cubin_base + kinfo->code_offset;

    const int N = 256;
    auto buf_a = gpu.gpu_malloc(N * 2);
    auto buf_b = gpu.gpu_malloc(N * 2);
    auto buf_y = gpu.gpu_malloc(N * 2);

    uint16_t* a = (uint16_t*)buf_a.cpu_ptr;
    uint16_t* b = (uint16_t*)buf_b.cpu_ptr;
    srand(42);
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

    // Test config 1: CUDA original (ONE_WORD, 32-bit payload)
    // DW[9] = 0x03: RELEASE0_ENABLE(1) + RELEASE_STRUCTURE_SIZE(1=ONE_WORD)
    launch_qmd_only(gpu, kernel_addr, kinfo->code_size, kinfo->reg_count,
                    kinfo->shared_mem_size, 1, cbuf.gpu_addr, kinfo->cbuf0_size,
                    0x00000003, "ONE_WORD+32bit (CUDA original)");

    // Test config 2: FOUR_WORDS, 64-bit payload
    // DW[9] = 0x1001: RELEASE0_ENABLE(1) + RELEASE_STRUCTURE_SIZE(0=FOUR_WORDS) + RELEASE_PAYLOAD64B(1)
    launch_qmd_only(gpu, kernel_addr, kinfo->code_size, kinfo->reg_count,
                    kinfo->shared_mem_size, 1, cbuf.gpu_addr, kinfo->cbuf0_size,
                    0x00001001, "FOUR_WORDS+64bit");

    // Test config 3: ONE_WORD, 64-bit payload
    // DW[9] = 0x1003
    launch_qmd_only(gpu, kernel_addr, kinfo->code_size, kinfo->reg_count,
                    kinfo->shared_mem_size, 1, cbuf.gpu_addr, kinfo->cbuf0_size,
                    0x00001003, "ONE_WORD+64bit");

    // Test config 4: FOUR_WORDS, 32-bit payload
    // DW[9] = 0x0001
    launch_qmd_only(gpu, kernel_addr, kinfo->code_size, kinfo->reg_count,
                    kinfo->shared_mem_size, 1, cbuf.gpu_addr, kinfo->cbuf0_size,
                    0x00000001, "FOUR_WORDS+32bit");

    return 0;
}
