// test_singlepb.cpp — Two kernel dispatches in a SINGLE pushbuffer entry
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

struct KernelArgs {
    GPU::GpuAlloc buf_a, buf_b, buf_y;
    GPU::GpuAlloc cbuf;
    GPU::GpuAlloc qmd_mem;
    int N;
    int seed;
};

void build_qmd(GPU& gpu, QMD& qmd, uint64_t kernel_addr, uint32_t code_size,
               uint32_t reg_count, uint64_t cbuf0_gpu, uint32_t cbuf0_size, uint32_t grid_x) {
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
}

int main() {
    fprintf(stderr, "=== test_singlepb: two dispatches in ONE pushbuffer ===\n\n");

    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) return 1;
    auto* kinfo = cubin.find_kernel("add_relu_kernel");
    if (!kinfo) return 1;

    GPU gpu;
    if (!gpu.init()) return 1;
    uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());
    uint64_t kernel_addr = cubin_base + kinfo->code_offset;

    const int N = 256;

    // Allocate buffers for BOTH dispatches upfront
    auto a1 = gpu.gpu_malloc(N*2), b1 = gpu.gpu_malloc(N*2), y1 = gpu.gpu_malloc(N*2);
    auto a2 = gpu.gpu_malloc(N*2), b2 = gpu.gpu_malloc(N*2), y2 = gpu.gpu_malloc(N*2);

    // Fill inputs
    uint16_t *pa1=(uint16_t*)a1.cpu_ptr, *pb1=(uint16_t*)b1.cpu_ptr, *py1=(uint16_t*)y1.cpu_ptr;
    uint16_t *pa2=(uint16_t*)a2.cpu_ptr, *pb2=(uint16_t*)b2.cpu_ptr, *py2=(uint16_t*)y2.cpu_ptr;
    for (int i=0;i<N;i++) { py1[i]=0xBEEF; py2[i]=0xDEAD; }
    srand(42);
    for (int i=0;i<N;i++) {
        pa1[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
        pb1[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
    }
    srand(99);
    for (int i=0;i<N;i++) {
        pa2[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
        pb2[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);
    }
    __sync_synchronize();

    // Prepare cbuf0 for both
    struct __attribute__((packed)) { uint64_t a,b,y; int32_t n; } args1, args2;
    args1 = {a1.gpu_addr, b1.gpu_addr, y1.gpu_addr, N};
    args2 = {a2.gpu_addr, b2.gpu_addr, y2.gpu_addr, N};
    auto cbuf1 = gpu.prepare_cbuf0(&args1, sizeof(args1), kinfo->cbuf0_size, kinfo->param_base, 256,1,1);
    auto cbuf2 = gpu.prepare_cbuf0(&args2, sizeof(args2), kinfo->cbuf0_size, kinfo->param_base, 256,1,1);

    // Build QMDs
    QMD qmd1, qmd2;
    build_qmd(gpu, qmd1, kernel_addr, kinfo->code_size, kinfo->reg_count,
              cbuf1.gpu_addr, kinfo->cbuf0_size, 1);
    build_qmd(gpu, qmd2, kernel_addr, kinfo->code_size, kinfo->reg_count,
              cbuf2.gpu_addr, kinfo->cbuf0_size, 1);

    auto qmem1 = gpu.gpu_malloc(4096);
    auto qmem2 = gpu.gpu_malloc(4096);
    memcpy(qmem1.cpu_ptr, &qmd1, sizeof(qmd1));
    memcpy(qmem2.cpu_ptr, &qmd2, sizeof(qmd2));
    __sync_synchronize();

    fprintf(stderr, "QMD1 at GPU 0x%lx, QMD2 at GPU 0x%lx\n",
            (unsigned long)qmem1.gpu_addr, (unsigned long)qmem2.gpu_addr);

    // Build ONE pushbuffer with TWO dispatches
    gpu.cmd_begin();

    // Preamble
    gpu.nvm(1, 0x0000, 1); gpu.nvm_data(0xCEC0);
    gpu.nvm(4, 0x0000, 1); gpu.nvm_data(0xCAB5);

    uint64_t lw = 0x729300000000ULL, sw = 0x729400000000ULL;
    gpu.nvm(1, 0x07B0, 2);
    gpu.nvm_data((uint32_t)(lw >> 32));
    gpu.nvm_data((uint32_t)(lw & 0xFFFFFFFF));
    gpu.nvm(1, 0x02A0, 2);
    gpu.nvm_data((uint32_t)(sw >> 32));
    gpu.nvm_data((uint32_t)(sw & 0xFFFFFFFF));

    // Dispatch 1
    gpu.nvm(1, 0x02B4, 1); gpu.nvm_data((uint32_t)(qmem1.gpu_addr >> 8));
    gpu.nvm(1, 0x02C0, 1); gpu.nvm_data(9);

    // Dispatch 2 (same pushbuffer, immediately after)
    gpu.nvm(1, 0x02B4, 1); gpu.nvm_data((uint32_t)(qmem2.gpu_addr >> 8));
    gpu.nvm(1, 0x02C0, 1); gpu.nvm_data(9);

    // HOST semaphore release (subchannel 0) IN THE SAME pushbuffer
    gpu.init_semaphore();
    gpu.sem_counter++;
    gpu.nvm(0, 0x005C, 5);
    gpu.nvm_data((uint32_t)(gpu.sem_alloc.gpu_addr & 0xFFFFFFFF));
    gpu.nvm_data((uint32_t)(gpu.sem_alloc.gpu_addr >> 32));
    gpu.nvm_data((uint32_t)(gpu.sem_counter & 0xFFFFFFFF));
    gpu.nvm_data((uint32_t)(gpu.sem_counter >> 32));
    gpu.nvm_data(0x01000001);  // RELEASE + 64BIT

    fprintf(stderr, "Pushbuffer: %u bytes, submitting as single GPFIFO entry\n",
            gpu.cmdq_offset - gpu.cmdq_base);

    gpu.submit_compute();

    // Poll semaphore
    volatile uint64_t* sem64 = (volatile uint64_t*)gpu.sem_alloc.cpu_ptr;
    int spins = 0;
    while (*sem64 < gpu.sem_counter) {
        __sync_synchronize();
        if (++spins > 100000000) {
            fprintf(stderr, "Semaphore timeout! sem64=0x%lx expected=0x%lx\n",
                    (unsigned long)*sem64, (unsigned long)gpu.sem_counter);
            break;
        }
    }
    fprintf(stderr, "Semaphore: %s after %d spins (sem=0x%lx)\n",
            spins <= 100000000 ? "OK" : "TIMEOUT", spins, (unsigned long)*sem64);
    __sync_synchronize();

    // Check results
    int beef1=0, ok1=0, beef2=0, ok2=0;
    for (int i=0;i<N;i++) {
        if (py1[i]==0xBEEF) beef1++;
        float exp1=fmaxf(fp16_to_fp32(pa1[i])+fp16_to_fp32(pb1[i]), 0.0f);
        if (abs((int)py1[i]-(int)fp32_to_fp16(exp1))<=1) ok1++;
        if (py2[i]==0xDEAD) beef2++;
        float exp2=fmaxf(fp16_to_fp32(pa2[i])+fp16_to_fp32(pb2[i]), 0.0f);
        if (abs((int)py2[i]-(int)fp32_to_fp16(exp2))<=1) ok2++;
    }
    fprintf(stderr, "Kernel 1: beef=%d correct=%d/%d (y[0]=0x%04x)\n", beef1, ok1, N, py1[0]);
    fprintf(stderr, "Kernel 2: beef=%d correct=%d/%d (y[0]=0x%04x)\n", beef2, ok2, N, py2[0]);

    return (ok1==N && ok2==N) ? 0 : 1;
}
