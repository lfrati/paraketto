// test_launch.cpp — Launch add_relu_kernel via cudaless ioctls, verify output
//
// This test:
//   1. Initializes GPU via direct ioctls (no CUDA)
//   2. Loads kernels.cubin (compiled by nvcc)
//   3. Allocates GPU memory for inputs/outputs
//   4. Fills inputs with test data via CPU mapping
//   5. Launches add_relu_kernel via QMD + SEND_PCAS
//   6. Waits for completion via semaphore polling
//   7. Reads output via CPU mapping and verifies against CPU reference

#include "gpu.h"
#include "cubin_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// FP16 helpers (manual, no CUDA headers)
static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127;
    uint32_t frac = x & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;      // inf
    if (exp < -14) return sign;               // zero/denorm
    return sign | ((exp + 15) << 10) | (frac >> 13);
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) { if (frac == 0) { float r; uint32_t v = sign; memcpy(&r, &v, 4); return r; }
        // denorm
        exp = 1;
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
    } else if (exp == 31) {
        float r; uint32_t v = sign | 0x7F800000 | (frac << 13); memcpy(&r, &v, 4); return r;
    }
    uint32_t v = sign | ((exp + 112) << 23) | (frac << 13);
    float r; memcpy(&r, &v, 4); return r;
}

int main() {
    fprintf(stderr, "=== test_launch: add_relu via cudaless ===\n\n");

    // Load CUBIN
    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) {
        fprintf(stderr, "FAILED: could not load kernels.cubin\n");
        return 1;
    }

    auto* kinfo = cubin.find_kernel("add_relu_kernel");
    if (!kinfo) {
        fprintf(stderr, "FAILED: add_relu_kernel not found in CUBIN\n");
        return 1;
    }
    fprintf(stderr, "Kernel: %s\n  code_size=%lu regs=%u smem=%u cbuf0=%u param_base=%u\n",
            kinfo->mangled_name.c_str(), (unsigned long)kinfo->code_size,
            kinfo->reg_count, kinfo->shared_mem_size, kinfo->cbuf0_size, kinfo->param_base);

    // Init GPU
    GPU gpu;
    if (!gpu.init()) {
        fprintf(stderr, "FAILED: GPU init\n");
        return 1;
    }

    // Upload CUBIN to GPU
    uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());
    uint64_t kernel_addr = cubin_base + kinfo->code_offset;
    fprintf(stderr, "CUBIN uploaded: base=0x%lx kernel_addr=0x%lx\n",
            (unsigned long)cubin_base, (unsigned long)kernel_addr);

    // Test parameters
    const int N = 4096;
    const size_t buf_bytes = N * sizeof(uint16_t);

    // Allocate GPU buffers
    auto buf_a = gpu.gpu_malloc(buf_bytes);
    auto buf_b = gpu.gpu_malloc(buf_bytes);
    auto buf_y = gpu.gpu_malloc(buf_bytes);

    // Fill inputs
    uint16_t* a = (uint16_t*)buf_a.cpu_ptr;
    uint16_t* b = (uint16_t*)buf_b.cpu_ptr;
    uint16_t* y = (uint16_t*)buf_y.cpu_ptr;
    // Fill output with 0xBEEF pattern to detect if kernel writes at all
    for (int i = 0; i < N; i++) y[i] = 0xBEEF;

    srand(42);
    for (int i = 0; i < N; i++) {
        float va = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        float vb = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        a[i] = fp32_to_fp16(va);
        b[i] = fp32_to_fp16(vb);
    }
    __sync_synchronize();

    // Prepare cbuf0 (kernel arguments)
    // add_relu_kernel(const half* a, const half* b, half* y, int n)
    struct __attribute__((packed)) {
        uint64_t a_ptr;
        uint64_t b_ptr;
        uint64_t y_ptr;
        int32_t  n;
    } args;
    args.a_ptr = buf_a.gpu_addr;
    args.b_ptr = buf_b.gpu_addr;
    args.y_ptr = buf_y.gpu_addr;
    args.n     = N;

    auto cbuf = gpu.prepare_cbuf0(&args, sizeof(args), kinfo->cbuf0_size, kinfo->param_base,
                                   256, 1, 1);  // block dims

    fprintf(stderr, "Launching kernel: grid=(%d,1,1) block=(256,1,1)\n",
            (N + 255) / 256);

    // Launch
    gpu.launch_kernel(
        kernel_addr, kinfo->code_size,
        kinfo->reg_count, kinfo->shared_mem_size,
        (N + 255) / 256, 1, 1,   // grid
        256, 1, 1,                // block
        cbuf.gpu_addr, kinfo->cbuf0_size
    );

    // Wait
    gpu.wait_kernel();

    // Debug: check semaphore and output
    volatile uint64_t* sem_debug = (volatile uint64_t*)gpu.sem_alloc.cpu_ptr;
    fprintf(stderr, "Semaphore value: 0x%lx (expected 0x%lx)\n",
            (unsigned long)*sem_debug, (unsigned long)gpu.sem_counter);
    // Check if output was modified at all
    int beef_count = 0;
    int zero_count = 0;
    for (int i = 0; i < 16; i++) {
        if (y[i] == 0xBEEF) beef_count++;
        if (y[i] == 0) zero_count++;
    }
    fprintf(stderr, "First 16 output words: beef=%d zeros=%d (sample: 0x%04x 0x%04x 0x%04x 0x%04x)\n",
            beef_count, zero_count, y[0], y[1], y[2], y[3]);

    // Verify
    __sync_synchronize();
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float fa = fp16_to_fp32(a[i]);
        float fb = fp16_to_fp32(b[i]);
        float expected = fmaxf(fa + fb, 0.0f);
        float got = fp16_to_fp32(y[i]);

        // Allow 1 ULP tolerance in FP16
        uint16_t expected_h = fp32_to_fp16(expected);
        int diff = abs((int)y[i] - (int)expected_h);
        if (diff > 1) {
            if (mismatches < 5)
                fprintf(stderr, "  MISMATCH i=%d: a=%.4f b=%.4f expected=%.4f got=%.4f (h: 0x%04x vs 0x%04x)\n",
                        i, fa, fb, expected, got, expected_h, y[i]);
            mismatches++;
        }
    }

    if (mismatches == 0)
        fprintf(stderr, "\n[PASS] add_relu_kernel  N=%d  0 mismatches\n", N);
    else
        fprintf(stderr, "\n[FAIL] add_relu_kernel  N=%d  %d mismatches\n", N, mismatches);

    return mismatches ? 1 : 0;
}
