// test_gpfifo_reset.cpp — Can we dispatch via a 2nd GPFIFO entry at all?
// Tests: dispatch+sem in entry 0, long wait, dispatch+sem in entry 1
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

int main() {
    fprintf(stderr, "=== test_gpfifo_reset ===\n\n");

    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) return 1;
    auto* kinfo = cubin.find_kernel("add_relu_kernel");
    if (!kinfo) return 1;

    GPU gpu;
    if (!gpu.init()) return 1;
    uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());
    uint64_t kernel_addr = cubin_base + kinfo->code_offset;
    gpu.init_semaphore();

    const int N = 256;

    // Pre-allocate everything for BOTH dispatches
    auto ba1=gpu.gpu_malloc(N*2), bb1=gpu.gpu_malloc(N*2), by1=gpu.gpu_malloc(N*2);
    auto ba2=gpu.gpu_malloc(N*2), bb2=gpu.gpu_malloc(N*2), by2=gpu.gpu_malloc(N*2);

    uint16_t *a1=(uint16_t*)ba1.cpu_ptr, *b1=(uint16_t*)bb1.cpu_ptr, *y1=(uint16_t*)by1.cpu_ptr;
    uint16_t *a2=(uint16_t*)ba2.cpu_ptr, *b2=(uint16_t*)bb2.cpu_ptr, *y2=(uint16_t*)by2.cpu_ptr;
    for(int i=0;i<N;i++){y1[i]=0xBEEF;y2[i]=0xDEAD;}
    srand(42);
    for(int i=0;i<N;i++){a1[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);b1[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);}
    srand(99);
    for(int i=0;i<N;i++){a2[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);b2[i]=fp32_to_fp16(((float)rand()/RAND_MAX)*4-2);}
    __sync_synchronize();

    struct __attribute__((packed)){uint64_t a,b,y;int32_t n;} args1,args2;
    args1={ba1.gpu_addr,bb1.gpu_addr,by1.gpu_addr,N};
    args2={ba2.gpu_addr,bb2.gpu_addr,by2.gpu_addr,N};
    auto cbuf1=gpu.prepare_cbuf0(&args1,sizeof(args1),kinfo->cbuf0_size,kinfo->param_base,256,1,1);
    auto cbuf2=gpu.prepare_cbuf0(&args2,sizeof(args2),kinfo->cbuf0_size,kinfo->param_base,256,1,1);

    auto build_qmd = [&](uint64_t cbuf_gpu) {
        QMD qmd={}; memset(&qmd,0,sizeof(qmd));
        qmd.dw[4]=0x013f0000; qmd.dw[9]=0; qmd.dw[10]=0x00190000;
        qmd.dw[12]=0x02070100; qmd.dw[14]=0x2f5003a4;
        qmd.dw[19]=0x80610000; qmd.dw[20]=0x00000008; qmd.dw[22]=0x04000000;
        qmd.dw[58]=0x00000011;
        uint64_t ps4=kernel_addr>>4;
        qmd.dw[32]=(uint32_t)ps4; qmd.dw[33]=(uint32_t)(ps4>>32)&0x1FFFFF;
        uint32_t regs=(kinfo->reg_count+1)&~1u; if(regs<16)regs=16;
        qmd.dw[34]=(1<<16)|256; qmd.dw[35]=1|(regs<<8)|(1<<17);
        qmd.dw[36]=8|(3<<11)|(0x1A<<17)|(5<<23);
        qmd.dw[39]=1; qmd.dw[40]=1; qmd.dw[41]=1;
        uint64_t cs6=cbuf_gpu>>6; uint32_t css4=(kinfo->cbuf0_size+15)/16;
        qmd.dw[42]=(uint32_t)cs6; qmd.dw[43]=((uint32_t)(cs6>>32)&0x7FFFF)|(css4<<19);
        qmd.dw[59]=(uint32_t)(kernel_addr>>8); qmd.dw[60]=(uint32_t)(kernel_addr>>40);
        auto qm=gpu.gpu_malloc(4096);
        memcpy(qm.cpu_ptr,&qmd,sizeof(qmd));
        __sync_synchronize();
        return qm;
    };

    auto qm1 = build_qmd(cbuf1.gpu_addr);
    auto qm2 = build_qmd(cbuf2.gpu_addr);

    auto dispatch_with_sem = [&](GPU::GpuAlloc& qm) {
        gpu.cmd_begin();
        if (!gpu.channel_setup_done) {
            gpu.nvm(1,0x0000,1); gpu.nvm_data(0xCEC0);
            gpu.nvm(4,0x0000,1); gpu.nvm_data(0xCAB5);
            gpu.channel_setup_done = true;
        }
        uint64_t lw=0x729300000000ULL, sw=0x729400000000ULL;
        gpu.nvm(1,0x07B0,2); gpu.nvm_data((uint32_t)(lw>>32)); gpu.nvm_data((uint32_t)lw);
        gpu.nvm(1,0x02A0,2); gpu.nvm_data((uint32_t)(sw>>32)); gpu.nvm_data((uint32_t)sw);
        gpu.nvm(1,0x02B4,1); gpu.nvm_data((uint32_t)(qm.gpu_addr>>8));
        gpu.nvm(1,0x02C0,1); gpu.nvm_data(9);
        // SEM in same PB
        gpu.sem_counter++;
        gpu.nvm(0,0x005C,5);
        gpu.nvm_data((uint32_t)(gpu.sem_alloc.gpu_addr&0xFFFFFFFF));
        gpu.nvm_data((uint32_t)(gpu.sem_alloc.gpu_addr>>32));
        gpu.nvm_data((uint32_t)(gpu.sem_counter&0xFFFFFFFF));
        gpu.nvm_data((uint32_t)(gpu.sem_counter>>32));
        gpu.nvm_data(0x01000001);
        gpu.submit_compute();
    };

    // Dispatch 1
    fprintf(stderr, "--- Dispatch 1 ---\n");
    dispatch_with_sem(qm1);
    volatile uint64_t* sem64 = (volatile uint64_t*)gpu.sem_alloc.cpu_ptr;
    int spins=0;
    while(*sem64<gpu.sem_counter && ++spins<100000000) __sync_synchronize();
    usleep(1000); // 1ms for kernel to finish
    __sync_synchronize();
    int ok1=0;
    for(int i=0;i<N;i++){float e=fmaxf(fp16_to_fp32(a1[i])+fp16_to_fp32(b1[i]),0.0f);if(abs((int)y1[i]-(int)fp32_to_fp16(e))<=1)ok1++;}
    fprintf(stderr, "  Kernel 1: %d/%d correct, spins=%d, sem=0x%lx\n", ok1, N, spins, (unsigned long)*sem64);

    // Wait 2 seconds (give compute engine time to fully reset)
    fprintf(stderr, "  Waiting 2 seconds before dispatch 2...\n");
    sleep(2);

    // Dispatch 2
    fprintf(stderr, "--- Dispatch 2 ---\n");
    dispatch_with_sem(qm2);
    spins=0;
    while(*sem64<gpu.sem_counter && ++spins<100000000) __sync_synchronize();
    usleep(1000);
    __sync_synchronize();
    int ok2=0;
    for(int i=0;i<N;i++){float e=fmaxf(fp16_to_fp32(a2[i])+fp16_to_fp32(b2[i]),0.0f);if(abs((int)y2[i]-(int)fp32_to_fp16(e))<=1)ok2++;}
    fprintf(stderr, "  Kernel 2: %d/%d correct, spins=%d, sem=0x%lx, y[0]=0x%04x\n",
            ok2, N, spins, (unsigned long)*sem64, y2[0]);

    fprintf(stderr, "\n%s\n", (ok1==N && ok2==N) ? "BOTH PASS" : "FAILED");
    return 0;
}
