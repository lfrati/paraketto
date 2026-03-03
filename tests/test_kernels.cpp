// test_kernels.cpp — Test all 23 cudaless-launched kernels against CPU references
//
// Zero NVIDIA runtime dependencies. Uses gpu.h + cubin_loader.h only.
//
// Architecture: All 23 kernel dispatches are batched into a SINGLE pushbuffer
// (one GPFIFO entry) because Blackwell's GPFIFO controller permanently stalls
// after processing any pushbuffer entry containing SEND_PCAS. Within a single
// pushbuffer, multiple SEND_PCAS dispatches work correctly.
//
// Build:
//   g++ -std=c++17 -O2 -I src -I third_party/nv-headers \
//       -I third_party/nv-headers/nvidia/generated -I third_party/nv-headers/uvm \
//       tests/test_kernels.cpp -o test_kernels -lm

#include "gpu.h"
#include "cubin_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <vector>
#include <functional>

// =========================================================================
// FP16 helpers (no CUDA headers)
// =========================================================================

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
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
    if (exp == 0) {
        if (frac == 0) { float r; uint32_t v = sign; memcpy(&r, &v, 4); return r; }
        exp = 1;
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
    } else if (exp == 31) {
        float r; uint32_t v = sign | 0x7F800000 | (frac << 13); memcpy(&r, &v, 4); return r;
    }
    uint32_t v = sign | ((exp + 112) << 23) | (frac << 13);
    float r; memcpy(&r, &v, 4); return r;
}

// =========================================================================
// Comparison helpers
// =========================================================================

static int compare_fp16(const uint16_t* got, const uint16_t* expected, int n,
                        int max_ulp, const char* name) {
    int mis = 0;
    for (int i = 0; i < n; i++) {
        int diff = abs((int)got[i] - (int)expected[i]);
        if (diff > max_ulp) {
            if (mis < 3)
                fprintf(stderr, "  %s[%d]: got=0x%04x(%.4f) exp=0x%04x(%.4f)\n",
                        name, i, got[i], fp16_to_fp32(got[i]),
                        expected[i], fp16_to_fp32(expected[i]));
            mis++;
        }
    }
    return mis;
}

static int compare_fp32(const float* got, const float* expected, int n,
                        float rtol, float atol, const char* name) {
    int mis = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > atol + rtol * fabsf(expected[i])) {
            if (mis < 3)
                fprintf(stderr, "  %s[%d]: got=%.6f exp=%.6f diff=%.2e\n",
                        name, i, got[i], expected[i], diff);
            mis++;
        }
    }
    return mis;
}

static int compare_int(const int32_t* got, const int32_t* expected, int n,
                       const char* name) {
    int mis = 0;
    for (int i = 0; i < n; i++) {
        if (got[i] != expected[i]) {
            if (mis < 3)
                fprintf(stderr, "  %s[%d]: got=%d exp=%d\n", name, i, got[i], expected[i]);
            mis++;
        }
    }
    return mis;
}

// =========================================================================
// Random fill helpers
// =========================================================================

static void fill_fp16(uint16_t* buf, int n, float lo, float hi) {
    for (int i = 0; i < n; i++)
        buf[i] = fp32_to_fp16(lo + (float)rand() / RAND_MAX * (hi - lo));
}

static void fill_fp32(float* buf, int n, float lo, float hi) {
    for (int i = 0; i < n; i++)
        buf[i] = lo + (float)rand() / RAND_MAX * (hi - lo);
}

// =========================================================================
// Launch helper (NO wait — appends to current pushbuffer)
// =========================================================================

static void launch_only(GPU& gpu, CubinKernel* k, uint64_t cubin_base,
                        const void* args, uint32_t args_size,
                        uint32_t gx, uint32_t gy, uint32_t gz,
                        uint32_t bx, uint32_t by, uint32_t bz) {
    auto cbuf = gpu.prepare_cbuf0(args, args_size, k->cbuf0_size, k->param_base,
                                  bx, by, bz);
    gpu.launch_kernel(cubin_base + k->code_offset, k->code_size,
                      k->reg_count, k->shared_mem_size,
                      gx, gy, gz, bx, by, bz,
                      cbuf.gpu_addr, k->cbuf0_size);
}

// =========================================================================
// Test setup struct — launch phase returns this; verify runs after wait
// =========================================================================

struct TestSetup {
    const char* name;
    int total;
    std::function<int()> verify;  // returns mismatches; nullptr = skip
};

// =========================================================================
// 1. add_relu_kernel
// =========================================================================

static TestSetup setup_add_relu(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("add_relu_kernel");
    if (!k) return {"add_relu_kernel", 0, nullptr};

    const int N = 4096;
    auto ba = gpu.gpu_malloc(N * 2), bb = gpu.gpu_malloc(N * 2), by = gpu.gpu_malloc(N * 2);
    auto* a = (uint16_t*)ba.cpu_ptr;
    auto* b = (uint16_t*)bb.cpu_ptr;
    auto* y = (uint16_t*)by.cpu_ptr;
    fill_fp16(a, N, -2, 2); fill_fp16(b, N, -2, 2); memset(y, 0, N * 2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t a, b, y; int32_t n; } args =
        {ba.gpu_addr, bb.gpu_addr, by.gpu_addr, N};
    launch_only(gpu, k, base, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);

    return {"add_relu_kernel", N, [a, b, y, N]() {
        std::vector<uint16_t> ref(N);
        for (int i = 0; i < N; i++)
            ref[i] = fp32_to_fp16(fmaxf(fp16_to_fp32(a[i]) + fp16_to_fp32(b[i]), 0.0f));
        return compare_fp16(y, ref.data(), N, 1, "add_relu");
    }};
}

// =========================================================================
// 2. cast_fp32_to_fp16_kernel
// =========================================================================

static TestSetup setup_cast_fp32_to_fp16(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("cast_fp32_to_fp16_kernel");
    if (!k) return {"cast_fp32_to_fp16_kernel", 0, nullptr};

    const int N = 4096;
    auto bx = gpu.gpu_malloc(N * 4), by = gpu.gpu_malloc(N * 2);
    auto* x = (float*)bx.cpu_ptr;
    auto* y = (uint16_t*)by.cpu_ptr;
    fill_fp32(x, N, -2, 2); memset(y, 0, N * 2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t x, y; int32_t n; } args =
        {bx.gpu_addr, by.gpu_addr, N};
    launch_only(gpu, k, base, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);

    return {"cast_fp32_to_fp16_kernel", N, [x, y, N]() {
        std::vector<uint16_t> ref(N);
        for (int i = 0; i < N; i++) ref[i] = fp32_to_fp16(x[i]);
        return compare_fp16(y, ref.data(), N, 1, "cast");
    }};
}

// =========================================================================
// 3. silu_inplace_kernel
// =========================================================================

static TestSetup setup_silu_inplace(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("silu_inplace_kernel");
    if (!k) return {"silu_inplace_kernel", 0, nullptr};

    const int N = 4096;
    auto bx = gpu.gpu_malloc(N * 2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    fill_fp16(x, N, -3, 3);
    std::vector<uint16_t> x_orig(x, x + N);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t x; int32_t n; } args = {bx.gpu_addr, N};
    launch_only(gpu, k, base, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);

    return {"silu_inplace_kernel", N, [x, x_orig, N]() {
        std::vector<uint16_t> ref(N);
        for (int i = 0; i < N; i++) {
            float v = fp16_to_fp32(x_orig[i]);
            ref[i] = fp32_to_fp16(v / (1.0f + expf(-v)));
        }
        return compare_fp16(x, ref.data(), N, 2, "silu");
    }};
}

// =========================================================================
// 4. residual_add_kernel
// =========================================================================

static TestSetup setup_residual_add(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("residual_add_kernel");
    if (!k) return {"residual_add_kernel", 0, nullptr};

    const int N = 4096;
    const float alpha = 0.75f;
    auto ba = gpu.gpu_malloc(N*2), bb = gpu.gpu_malloc(N*2), by = gpu.gpu_malloc(N*2);
    auto* a = (uint16_t*)ba.cpu_ptr;
    auto* b = (uint16_t*)bb.cpu_ptr;
    auto* y = (uint16_t*)by.cpu_ptr;
    fill_fp16(a, N, -1, 1); fill_fp16(b, N, -1, 1); memset(y, 0, N*2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t a, b, y; int32_t n; float alpha; } args =
        {ba.gpu_addr, bb.gpu_addr, by.gpu_addr, N, alpha};
    launch_only(gpu, k, base, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);

    return {"residual_add_kernel", N, [a, b, y, N, alpha]() {
        std::vector<uint16_t> ref(N);
        for (int i = 0; i < N; i++)
            ref[i] = fp32_to_fp16(fp16_to_fp32(a[i]) + alpha * fp16_to_fp32(b[i]));
        return compare_fp16(y, ref.data(), N, 1, "res_add");
    }};
}

// =========================================================================
// 5. glu_kernel
// =========================================================================

static TestSetup setup_glu(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("glu_kernel");
    if (!k) return {"glu_kernel", 0, nullptr};

    const int N = 32, D = 64;
    int total = N * D;
    auto bx = gpu.gpu_malloc(N * 2 * D * 2), by = gpu.gpu_malloc(total * 2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    auto* y = (uint16_t*)by.cpu_ptr;
    fill_fp16(x, N * 2 * D, -2, 2); memset(y, 0, total * 2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t x, y; int32_t N, D; } args =
        {bx.gpu_addr, by.gpu_addr, N, D};
    launch_only(gpu, k, base, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);

    return {"glu_kernel", total, [x, y, N, D, total]() {
        std::vector<uint16_t> ref(total);
        for (int idx = 0; idx < total; idx++) {
            int row = idx / D, col = idx % D;
            float a = fp16_to_fp32(x[row * 2 * D + col]);
            float b = fp16_to_fp32(x[row * 2 * D + D + col]);
            ref[idx] = fp32_to_fp16(a * (1.0f / (1.0f + expf(-b))));
        }
        return compare_fp16(y, ref.data(), total, 2, "glu");
    }};
}

// =========================================================================
// 6. bias_relu_nchw_kernel (in-place)
// =========================================================================

static TestSetup setup_bias_relu_nchw(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("bias_relu_nchw_kernel");
    if (!k) return {"bias_relu_nchw_kernel", 0, nullptr};

    const int C = 16, spatial = 64;
    int total = C * spatial;
    auto bx = gpu.gpu_malloc(total * 2), bb = gpu.gpu_malloc(C * 2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    auto* bias = (uint16_t*)bb.cpu_ptr;
    fill_fp16(x, total, -2, 2); fill_fp16(bias, C, -0.5, 0.5);
    std::vector<uint16_t> x_orig(x, x + total);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t x, bias; int32_t C, spatial; } args =
        {bx.gpu_addr, bb.gpu_addr, C, spatial};
    launch_only(gpu, k, base, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);

    return {"bias_relu_nchw_kernel", total, [x, bias, x_orig, C, spatial, total]() {
        std::vector<uint16_t> ref(total);
        for (int i = 0; i < total; i++) {
            int c = i / spatial;
            float val = fp16_to_fp32(x_orig[i]) + fp16_to_fp32(bias[c]);
            ref[i] = fp32_to_fp16(fmaxf(val, 0.0f));
        }
        return compare_fp16(x, ref.data(), total, 1, "bias_relu");
    }};
}

// =========================================================================
// 7. bias_add_kernel (in-place)
// =========================================================================

static TestSetup setup_bias_add(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("bias_add_kernel");
    if (!k) return {"bias_add_kernel", 0, nullptr};

    const int rows = 32, cols = 64;
    int total = rows * cols;
    auto bx = gpu.gpu_malloc(total * 2), bb = gpu.gpu_malloc(cols * 2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    auto* bias = (uint16_t*)bb.cpu_ptr;
    fill_fp16(x, total, -1, 1); fill_fp16(bias, cols, -0.5, 0.5);
    std::vector<uint16_t> x_orig(x, x + total);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t x, bias; int32_t rows, cols; } args =
        {bx.gpu_addr, bb.gpu_addr, rows, cols};
    launch_only(gpu, k, base, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);

    return {"bias_add_kernel", total, [x, bias, x_orig, cols, total]() {
        std::vector<uint16_t> ref(total);
        for (int i = 0; i < total; i++) {
            int j = i % cols;
            ref[i] = fp32_to_fp16(fp16_to_fp32(x_orig[i]) + fp16_to_fp32(bias[j]));
        }
        return compare_fp16(x, ref.data(), total, 1, "bias_add");
    }};
}

// =========================================================================
// 8. lstm_cell_kernel
// =========================================================================

static TestSetup setup_lstm_cell(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("lstm_cell_kernel");
    if (!k) return {"lstm_cell_kernel", 0, nullptr};

    const int H = 256;
    auto bg = gpu.gpu_malloc(4*H*2), bc = gpu.gpu_malloc(H*2);
    auto bh = gpu.gpu_malloc(H*2), bco = gpu.gpu_malloc(H*2);
    auto* gates = (uint16_t*)bg.cpu_ptr;
    auto* c_prev = (uint16_t*)bc.cpu_ptr;
    auto* h_out = (uint16_t*)bh.cpu_ptr;
    auto* c_out = (uint16_t*)bco.cpu_ptr;
    fill_fp16(gates, 4*H, -2, 2); fill_fp16(c_prev, H, -1, 1);
    memset(h_out, 0, H*2); memset(c_out, 0, H*2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t gates, c_prev, h_out, c_out; int32_t H; } args =
        {bg.gpu_addr, bc.gpu_addr, bh.gpu_addr, bco.gpu_addr, H};
    launch_only(gpu, k, base, &args, sizeof(args), (H+255)/256, 1, 1, 256, 1, 1);

    return {"lstm_cell_kernel", 2*H, [gates, c_prev, h_out, c_out, H]() {
        std::vector<uint16_t> ref_h(H), ref_c(H);
        for (int i = 0; i < H; i++) {
            float gi = fp16_to_fp32(gates[i]);
            float go = fp16_to_fp32(gates[H+i]);
            float gf = fp16_to_fp32(gates[2*H+i]);
            float gg = fp16_to_fp32(gates[3*H+i]);
            float ig = 1.0f / (1.0f + expf(-gi));
            float og = 1.0f / (1.0f + expf(-go));
            float fg = 1.0f / (1.0f + expf(-gf));
            float cg = tanhf(gg);
            float c = fg * fp16_to_fp32(c_prev[i]) + ig * cg;
            float h = og * tanhf(c);
            ref_c[i] = fp32_to_fp16(c);
            ref_h[i] = fp32_to_fp16(h);
        }
        return compare_fp16(h_out, ref_h.data(), H, 2, "lstm_h")
             + compare_fp16(c_out, ref_c.data(), H, 2, "lstm_c");
    }};
}

// =========================================================================
// 9. embed_concat_kernel (has 4B padding after int idx)
// =========================================================================

static TestSetup setup_embed_concat(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("embed_concat_kernel");
    if (!k) return {"embed_concat_kernel", 0, nullptr};

    const int D = 128, vocab = 16, idx = 5;
    auto bt = gpu.gpu_malloc(vocab*D*2), bh = gpu.gpu_malloc(D*2), bo = gpu.gpu_malloc(2*D*2);
    auto* table = (uint16_t*)bt.cpu_ptr;
    auto* h = (uint16_t*)bh.cpu_ptr;
    auto* out = (uint16_t*)bo.cpu_ptr;
    fill_fp16(table, vocab*D, -1, 1); fill_fp16(h, D, -1, 1); memset(out, 0, 2*D*2);
    __sync_synchronize();

    struct __attribute__((packed)) {
        uint64_t table; int32_t idx; int32_t _pad;
        uint64_t h, out; int32_t D;
    } args = {bt.gpu_addr, idx, 0, bh.gpu_addr, bo.gpu_addr, D};
    launch_only(gpu, k, base, &args, sizeof(args), 1, 1, 1, 256, 1, 1);

    return {"embed_concat_kernel", 2*D, [table, h, out, D, idx]() {
        std::vector<uint16_t> ref(2*D);
        memcpy(ref.data(), &table[idx*D], D*2);
        memcpy(ref.data()+D, h, D*2);
        return compare_fp16(out, ref.data(), 2*D, 0, "embed_concat");
    }};
}

// =========================================================================
// 10. concat_vectors_kernel
// =========================================================================

static TestSetup setup_concat_vectors(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("concat_vectors_kernel");
    if (!k) return {"concat_vectors_kernel", 0, nullptr};

    const int D = 128;
    auto ba = gpu.gpu_malloc(D*2), bb = gpu.gpu_malloc(D*2), bo = gpu.gpu_malloc(2*D*2);
    auto* a = (uint16_t*)ba.cpu_ptr;
    auto* b = (uint16_t*)bb.cpu_ptr;
    auto* out = (uint16_t*)bo.cpu_ptr;
    fill_fp16(a, D, -1, 1); fill_fp16(b, D, -1, 1); memset(out, 0, 2*D*2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t a, b, out; int32_t D; } args =
        {ba.gpu_addr, bb.gpu_addr, bo.gpu_addr, D};
    launch_only(gpu, k, base, &args, sizeof(args), 1, 1, 1, 256, 1, 1);

    return {"concat_vectors_kernel", 2*D, [a, b, out, D]() {
        std::vector<uint16_t> ref(2*D);
        memcpy(ref.data(), a, D*2);
        memcpy(ref.data()+D, b, D*2);
        return compare_fp16(out, ref.data(), 2*D, 0, "concat_vec");
    }};
}

// =========================================================================
// 11. reshape_chw_to_hcw_kernel
// =========================================================================

static TestSetup setup_reshape_chw_to_hcw(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("reshape_chw_to_hcw_kernel");
    if (!k) return {"reshape_chw_to_hcw_kernel", 0, nullptr};

    const int C = 4, H = 8, W = 8;
    int total = C*H*W;
    auto bi = gpu.gpu_malloc(total*2), bo = gpu.gpu_malloc(total*2);
    auto* in = (uint16_t*)bi.cpu_ptr;
    auto* out = (uint16_t*)bo.cpu_ptr;
    fill_fp16(in, total, -1, 1); memset(out, 0, total*2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t in_, out_; int32_t C, H, W; } args =
        {bi.gpu_addr, bo.gpu_addr, C, H, W};
    launch_only(gpu, k, base, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);

    return {"reshape_chw_to_hcw_kernel", total, [in, out, C, H, W, total]() {
        std::vector<uint16_t> ref(total);
        for (int idx = 0; idx < total; idx++) {
            int h = idx / (C*W), cw = idx % (C*W);
            int c = cw / W, w = cw % W;
            ref[idx] = in[c*H*W + h*W + w];
        }
        return compare_fp16(out, ref.data(), total, 0, "reshape");
    }};
}

// =========================================================================
// 12. transpose_0213_kernel
// =========================================================================

static TestSetup setup_transpose_0213(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("transpose_0213_kernel");
    if (!k) return {"transpose_0213_kernel", 0, nullptr};

    const int A = 4, B = 8, C = 64;
    int total = A*B*C;
    auto bi = gpu.gpu_malloc(total*2), bo = gpu.gpu_malloc(total*2);
    auto* in = (uint16_t*)bi.cpu_ptr;
    auto* out = (uint16_t*)bo.cpu_ptr;
    fill_fp16(in, total, -1, 1); memset(out, 0, total*2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t in_, out_; int32_t A, B, C; } args =
        {bi.gpu_addr, bo.gpu_addr, A, B, C};
    launch_only(gpu, k, base, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);

    return {"transpose_0213_kernel", total, [in, out, A, B, C, total]() {
        std::vector<uint16_t> ref(total);
        for (int idx = 0; idx < total; idx++) {
            int a = idx / (B*C), rem = idx % (B*C);
            int b = rem / C, c = rem % C;
            ref[b*A*C + a*C + c] = in[idx];
        }
        return compare_fp16(out, ref.data(), total, 0, "t0213");
    }};
}

// =========================================================================
// 13. depthwise_conv1d_k9_silu_kernel
// =========================================================================

static TestSetup setup_depthwise_conv1d(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("depthwise_conv1d_k9_silu_kernel");
    if (!k) return {"depthwise_conv1d_k9_silu_kernel", 0, nullptr};

    const int T = 32, C = 16;
    int total = T * C;
    auto bx = gpu.gpu_malloc(total*2), bw = gpu.gpu_malloc(C*9*2);
    auto bb = gpu.gpu_malloc(C*2), by = gpu.gpu_malloc(total*2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    auto* w = (uint16_t*)bw.cpu_ptr;
    auto* bias = (uint16_t*)bb.cpu_ptr;
    auto* y = (uint16_t*)by.cpu_ptr;
    fill_fp16(x, total, -1, 1); fill_fp16(w, C*9, -0.5, 0.5);
    fill_fp16(bias, C, -0.3, 0.3); memset(y, 0, total*2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t x, w, b, y; int32_t T, C; } args =
        {bx.gpu_addr, bw.gpu_addr, bb.gpu_addr, by.gpu_addr, T, C};
    launch_only(gpu, k, base, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);

    return {"depthwise_conv1d_k9_silu_kernel", total, [x, w, bias, y, T, C, total]() {
        std::vector<uint16_t> ref(total);
        for (int idx = 0; idx < total; idx++) {
            int t = idx / C, c = idx % C;
            float sum = 0.0f;
            for (int kk = 0; kk < 9; kk++) {
                int ti = t + kk - 4;
                if (ti >= 0 && ti < T)
                    sum += fp16_to_fp32(x[ti*C+c]) * fp16_to_fp32(w[c*9+kk]);
            }
            sum += fp16_to_fp32(bias[c]);
            ref[idx] = fp32_to_fp16(sum / (1.0f + expf(-sum)));
        }
        return compare_fp16(y, ref.data(), total, 2, "dw_conv");
    }};
}

// =========================================================================
// 14. conv2d_kernel
// =========================================================================

static TestSetup setup_conv2d(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("conv2d_kernel");
    if (!k) return {"conv2d_kernel", 0, nullptr};

    const int C_in = 4, H_in = 8, W_in = 8;
    const int C_out = 8, kH = 3, kW = 3, stride = 1, pad = 1, groups = 1;
    int c_per_group = C_in / groups;
    int H_out = (H_in + 2*pad - kH) / stride + 1;
    int W_out = (W_in + 2*pad - kW) / stride + 1;
    int in_sz = C_in*H_in*W_in, out_sz = C_out*H_out*W_out;
    int w_sz = C_out * c_per_group * kH * kW;

    auto bi = gpu.gpu_malloc(in_sz*2), bw = gpu.gpu_malloc(w_sz*2);
    auto bb = gpu.gpu_malloc(C_out*2), bo = gpu.gpu_malloc(out_sz*2);
    auto* input = (uint16_t*)bi.cpu_ptr;
    auto* weight = (uint16_t*)bw.cpu_ptr;
    auto* bias = (uint16_t*)bb.cpu_ptr;
    auto* output = (uint16_t*)bo.cpu_ptr;
    fill_fp16(input, in_sz, -0.5, 0.5);
    fill_fp16(weight, w_sz, -0.3, 0.3);
    fill_fp16(bias, C_out, -0.1, 0.1);
    memset(output, 0, out_sz*2);
    __sync_synchronize();

    struct __attribute__((packed)) {
        uint64_t input, weight, bias, output;
        int32_t C_in, H_in, W_in, C_out, H_out, W_out;
        int32_t kH, kW, stride, pad, groups, c_per_group;
    } args = {bi.gpu_addr, bw.gpu_addr, bb.gpu_addr, bo.gpu_addr,
              C_in, H_in, W_in, C_out, H_out, W_out, kH, kW, stride, pad, groups, c_per_group};
    launch_only(gpu, k, base, &args, sizeof(args), (out_sz+255)/256, 1, 1, 256, 1, 1);

    return {"conv2d_kernel", out_sz, [input, weight, bias, output,
            C_in, H_in, W_in, C_out, H_out, W_out, kH, kW, stride, pad, groups, c_per_group, out_sz]() {
        std::vector<uint16_t> ref(out_sz);
        for (int idx = 0; idx < out_sz; idx++) {
            int oc = idx / (H_out * W_out);
            int rem = idx % (H_out * W_out);
            int oh = rem / W_out, ow = rem % W_out;
            int group = oc / (C_out / groups);
            int ic_start = group * c_per_group;
            float sum = 0.0f;
            for (int ic = 0; ic < c_per_group; ic++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw_i = 0; kw_i < kW; kw_i++) {
                        int ih = oh*stride + kh - pad, iw = ow*stride + kw_i - pad;
                        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)
                            sum += fp16_to_fp32(input[(ic_start+ic)*H_in*W_in + ih*W_in + iw])
                                 * fp16_to_fp32(weight[oc*c_per_group*kH*kW + ic*kH*kW + kh*kW + kw_i]);
                    }
            sum += fp16_to_fp32(bias[oc]);
            ref[idx] = fp32_to_fp16(sum);
        }
        return compare_fp16(output, ref.data(), out_sz, 4, "conv2d");
    }};
}

// =========================================================================
// 15. im2col_2d_kernel
// =========================================================================

static TestSetup setup_im2col(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("im2col_2d_kernel");
    if (!k) return {"im2col_2d_kernel", 0, nullptr};

    const int C_in = 4, H_in = 8, W_in = 8;
    const int kH = 3, kW = 3, stride = 1, pad = 1;
    int H_out = (H_in + 2*pad - kH) / stride + 1;
    int W_out = (W_in + 2*pad - kW) / stride + 1;
    int col_rows = C_in * kH * kW;
    int col_cols = H_out * W_out;
    int total = col_rows * col_cols;

    auto bi = gpu.gpu_malloc(C_in*H_in*W_in*2), bo = gpu.gpu_malloc(total*2);
    auto* input = (uint16_t*)bi.cpu_ptr;
    auto* col = (uint16_t*)bo.cpu_ptr;
    fill_fp16(input, C_in*H_in*W_in, -1, 1); memset(col, 0, total*2);
    __sync_synchronize();

    struct __attribute__((packed)) {
        uint64_t input, col;
        int32_t C_in, H_in, W_in, kH, kW, stride, pad, H_out, W_out;
    } args = {bi.gpu_addr, bo.gpu_addr, C_in, H_in, W_in, kH, kW, stride, pad, H_out, W_out};
    launch_only(gpu, k, base, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);

    return {"im2col_2d_kernel", total, [input, col, C_in, H_in, W_in, kH, kW, stride, pad,
            H_out, W_out, col_rows, col_cols, total]() {
        std::vector<uint16_t> ref(total);
        for (int idx = 0; idx < total; idx++) {
            int spatial = idx % col_cols, patch = idx / col_cols;
            int oh = spatial / W_out, ow = spatial % W_out;
            int ic = patch / (kH * kW), kk = patch % (kH * kW);
            int kh = kk / kW, kw_i = kk % kW;
            int ih = oh * stride + kh - pad, iw = ow * stride + kw_i - pad;
            ref[idx] = (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in)
                       ? input[ic*H_in*W_in + ih*W_in + iw] : fp32_to_fp16(0.0f);
        }
        return compare_fp16(col, ref.data(), total, 0, "im2col");
    }};
}

// =========================================================================
// 16. generate_pos_encoding_kernel
// =========================================================================

static TestSetup setup_pos_encoding(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("generate_pos_encoding_kernel");
    if (!k) return {"generate_pos_encoding_kernel", 0, nullptr};

    const int T = 16, d_model = 64;
    int len = 2*T - 1;
    int half_d = d_model / 2;
    int total_elems = len * d_model;
    int total_work = len * half_d;

    auto bo = gpu.gpu_malloc(total_elems * 2);
    auto* out = (uint16_t*)bo.cpu_ptr;
    memset(out, 0, total_elems * 2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t output; int32_t T, d_model; } args =
        {bo.gpu_addr, T, d_model};
    launch_only(gpu, k, base, &args, sizeof(args), (total_work+255)/256, 1, 1, 256, 1, 1);

    return {"generate_pos_encoding_kernel", total_elems,
            [out, T, d_model, len, half_d, total_elems, total_work]() {
        std::vector<uint16_t> ref(total_elems, 0);
        for (int idx = 0; idx < total_work; idx++) {
            int p = idx / half_d, i = idx % half_d;
            float pos = (float)(T - 1 - p);
            float div_term = expf(-(float)(2*i) * logf(10000.0f) / d_model);
            float angle = pos * div_term;
            ref[p * d_model + 2*i]     = fp32_to_fp16(sinf(angle));
            ref[p * d_model + 2*i + 1] = fp32_to_fp16(cosf(angle));
        }
        return compare_fp16(out, ref.data(), total_elems, 2, "pos_enc");
    }};
}

// =========================================================================
// 17. split_transpose_qkv_bias_kernel
// =========================================================================

static TestSetup setup_split_transpose_qkv(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("split_transpose_qkv_bias_kernel");
    if (!k) return {"split_transpose_qkv_bias_kernel", 0, nullptr};

    const int T = 16, heads = 4, hd = 32;
    int D = heads * hd;
    int in_sz = T * 3 * D;
    int out_sz = heads * T * hd;

    auto bin = gpu.gpu_malloc(in_sz*2);
    auto bbu = gpu.gpu_malloc(D*2), bbv = gpu.gpu_malloc(D*2);
    auto bqu = gpu.gpu_malloc(out_sz*2), bqv = gpu.gpu_malloc(out_sz*2);
    auto bk = gpu.gpu_malloc(out_sz*2), bv = gpu.gpu_malloc(out_sz*2);

    auto* inp = (uint16_t*)bin.cpu_ptr;
    auto* bu = (uint16_t*)bbu.cpu_ptr;
    auto* bv_ = (uint16_t*)bbv.cpu_ptr;
    auto* qu = (uint16_t*)bqu.cpu_ptr;
    auto* qv = (uint16_t*)bqv.cpu_ptr;
    auto* ko = (uint16_t*)bk.cpu_ptr;
    auto* vo = (uint16_t*)bv.cpu_ptr;
    fill_fp16(inp, in_sz, -1, 1); fill_fp16(bu, D, -0.5, 0.5); fill_fp16(bv_, D, -0.5, 0.5);
    memset(qu, 0, out_sz*2); memset(qv, 0, out_sz*2);
    memset(ko, 0, out_sz*2); memset(vo, 0, out_sz*2);
    __sync_synchronize();

    int total_work = T * D;
    struct __attribute__((packed)) {
        uint64_t in_, bu_, bv_, qu_, qv_, k_, v_;
        int32_t T, heads, hd;
    } args = {bin.gpu_addr, bbu.gpu_addr, bbv.gpu_addr,
              bqu.gpu_addr, bqv.gpu_addr, bk.gpu_addr, bv.gpu_addr,
              T, heads, hd};
    launch_only(gpu, k, base, &args, sizeof(args), (total_work+255)/256, 1, 1, 256, 1, 1);

    return {"split_transpose_qkv_bias_kernel", 4*out_sz,
            [inp, bu, bv_, qu, qv, ko, vo, T, heads, hd, D, out_sz, total_work]() {
        std::vector<uint16_t> rqu(out_sz), rqv(out_sz), rk(out_sz), rv(out_sz);
        for (int idx = 0; idx < total_work; idx++) {
            int t = idx / D, rem = idx % D;
            int h = rem / hd, d = rem % hd;
            int in_row = t * 3 * D;
            int oi = h * T * hd + t * hd + d;
            float q = fp16_to_fp32(inp[in_row + rem]);
            rqu[oi] = fp32_to_fp16(q + fp16_to_fp32(bu[h*hd+d]));
            rqv[oi] = fp32_to_fp16(q + fp16_to_fp32(bv_[h*hd+d]));
            rk[oi] = inp[in_row + D + rem];
            rv[oi] = inp[in_row + 2*D + rem];
        }
        return compare_fp16(qu, rqu.data(), out_sz, 1, "qkv_qu")
             + compare_fp16(qv, rqv.data(), out_sz, 1, "qkv_qv")
             + compare_fp16(ko, rk.data(), out_sz, 0, "qkv_k")
             + compare_fp16(vo, rv.data(), out_sz, 0, "qkv_v");
    }};
}

// =========================================================================
// 18. layer_norm_kernel
// =========================================================================

static TestSetup setup_layer_norm(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("layer_norm_kernel");
    if (!k) return {"layer_norm_kernel", 0, nullptr};

    const int N = 16, D = 128;
    const float eps = 1e-5f;
    int total = N * D;
    auto bx = gpu.gpu_malloc(total*2), bg = gpu.gpu_malloc(D*2);
    auto bb = gpu.gpu_malloc(D*2), by = gpu.gpu_malloc(total*2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    auto* gamma = (uint16_t*)bg.cpu_ptr;
    auto* beta = (uint16_t*)bb.cpu_ptr;
    auto* y = (uint16_t*)by.cpu_ptr;
    fill_fp16(x, total, -1, 1); fill_fp16(gamma, D, 0.5, 1.5); fill_fp16(beta, D, -0.5, 0.5);
    memset(y, 0, total*2);
    __sync_synchronize();

    struct __attribute__((packed)) {
        uint64_t x, gamma, beta, y; int32_t N, D; float eps;
    } args = {bx.gpu_addr, bg.gpu_addr, bb.gpu_addr, by.gpu_addr, N, D, eps};
    int block = ((D + 31) / 32) * 32;
    if (block > 1024) block = 1024;
    if (block < 32) block = 32;
    launch_only(gpu, k, base, &args, sizeof(args), N, 1, 1, block, 1, 1);

    return {"layer_norm_kernel", total, [x, gamma, beta, y, N, D, eps, total]() {
        std::vector<uint16_t> ref(total);
        for (int row = 0; row < N; row++) {
            float sum = 0, sum2 = 0;
            for (int i = 0; i < D; i++) {
                float v = fp16_to_fp32(x[row*D+i]);
                sum += v; sum2 += v*v;
            }
            float mean = sum / D;
            float var = sum2 / D - mean * mean;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int i = 0; i < D; i++) {
                float v = fp16_to_fp32(x[row*D+i]);
                float g = fp16_to_fp32(gamma[i]);
                float b = fp16_to_fp32(beta[i]);
                ref[row*D+i] = fp32_to_fp16((v - mean) * inv_std * g + b);
            }
        }
        return compare_fp16(y, ref.data(), total, 3, "ln");
    }};
}

// =========================================================================
// 19. residual_add_layer_norm_kernel (has 4B padding after float alpha)
// =========================================================================

static TestSetup setup_residual_add_layer_norm(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("residual_add_layer_norm_kernel");
    if (!k) return {"residual_add_layer_norm_kernel", 0, nullptr};

    const int N = 16, D = 128;
    const float alpha = 0.8f, eps = 1e-5f;
    int total = N * D;

    auto bx = gpu.gpu_malloc(total*2), bd = gpu.gpu_malloc(total*2);
    auto bg = gpu.gpu_malloc(D*2), bb = gpu.gpu_malloc(D*2), bln = gpu.gpu_malloc(total*2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    auto* delta = (uint16_t*)bd.cpu_ptr;
    auto* gamma = (uint16_t*)bg.cpu_ptr;
    auto* beta = (uint16_t*)bb.cpu_ptr;
    auto* ln_out = (uint16_t*)bln.cpu_ptr;
    fill_fp16(x, total, -1, 1); fill_fp16(delta, total, -0.5, 0.5);
    fill_fp16(gamma, D, 0.5, 1.5); fill_fp16(beta, D, -0.5, 0.5);
    memset(ln_out, 0, total*2);
    std::vector<uint16_t> x_orig(x, x + total);
    __sync_synchronize();

    struct __attribute__((packed)) {
        uint64_t x, delta; float alpha; int32_t _pad;
        uint64_t gamma, beta, ln_out;
        int32_t N, D; float eps;
    } args = {bx.gpu_addr, bd.gpu_addr, alpha, 0,
              bg.gpu_addr, bb.gpu_addr, bln.gpu_addr, N, D, eps};

    int block = ((D + 31) / 32) * 32;
    if (block > 1024) block = 1024;
    if (block < 32) block = 32;
    launch_only(gpu, k, base, &args, sizeof(args), N, 1, 1, block, 1, 1);

    return {"residual_add_layer_norm_kernel", total,
            [x, delta, gamma, beta, ln_out, x_orig, alpha, eps, N, D, total]() {
        std::vector<float> x_upd(total);
        for (int i = 0; i < total; i++)
            x_upd[i] = fp16_to_fp32(x_orig[i]) + alpha * fp16_to_fp32(delta[i]);

        std::vector<uint16_t> ref(total);
        for (int row = 0; row < N; row++) {
            float sum = 0, sum2 = 0;
            for (int i = 0; i < D; i++) {
                float v = fp16_to_fp32(fp32_to_fp16(x_upd[row*D+i]));
                sum += v; sum2 += v*v;
            }
            float mean = sum / D;
            float var = sum2 / D - mean * mean;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int i = 0; i < D; i++) {
                float v = fp16_to_fp32(fp32_to_fp16(x_upd[row*D+i]));
                float g = fp16_to_fp32(gamma[i]);
                float b = fp16_to_fp32(beta[i]);
                ref[row*D+i] = fp32_to_fp16((v - mean) * inv_std * g + b);
            }
        }
        // Use float-domain comparison: layer norm accumulation order differs on GPU
        int mis = 0;
        for (int i = 0; i < total; i++) {
            float g = fp16_to_fp32(ln_out[i]);
            float e = fp16_to_fp32(ref[i]);
            float diff = fabsf(g - e);
            if (diff > 0.01f + 0.05f * fabsf(e)) {
                if (mis < 3)
                    fprintf(stderr, "  res_ln[%d]: got=%.4f exp=%.4f diff=%.4f\n", i, g, e, diff);
                mis++;
            }
        }
        return mis;
    }};
}

// =========================================================================
// 20. fused_score_softmax_kernel
// =========================================================================

static TestSetup setup_fused_score_softmax(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("fused_score_softmax_kernel");
    if (!k) return {"fused_score_softmax_kernel", 0, nullptr};

    const int heads = 4, T = 16;
    const float scale = 1.0f / sqrtf(32.0f);
    int W = 2*T - 1;
    int rows = heads * T;
    int content_sz = rows * T;
    int pos_sz = heads * T * W;

    auto bc = gpu.gpu_malloc(content_sz*2), bp = gpu.gpu_malloc(pos_sz*2);
    auto bo = gpu.gpu_malloc(content_sz*2);
    auto* content = (uint16_t*)bc.cpu_ptr;
    auto* pos = (uint16_t*)bp.cpu_ptr;
    auto* output = (uint16_t*)bo.cpu_ptr;
    fill_fp16(content, content_sz, -1, 1);
    fill_fp16(pos, pos_sz, -1, 1);
    memset(output, 0, content_sz*2);
    __sync_synchronize();

    struct __attribute__((packed)) {
        uint64_t content, pos, output; int32_t T; float scale;
    } args = {bc.gpu_addr, bp.gpu_addr, bo.gpu_addr, T, scale};

    int block = ((T + 31) / 32) * 32;
    if (block > 1024) block = 1024;
    if (block < 32) block = 32;
    launch_only(gpu, k, base, &args, sizeof(args), rows, 1, 1, block, 1, 1);

    return {"fused_score_softmax_kernel", content_sz,
            [content, pos, output, heads, T, scale, W, rows, content_sz]() {
        std::vector<uint16_t> ref(content_sz);
        for (int row = 0; row < rows; row++) {
            int h = row / T, t = row % T;
            std::vector<float> scores(T);
            float max_val = -FLT_MAX;
            for (int j = 0; j < T; j++) {
                int src_col = j + T - 1 - t;
                float c = fp16_to_fp32(content[row*T+j]);
                float p = fp16_to_fp32(pos[h*T*W + t*W + src_col]);
                scores[j] = (c + p) * scale;
                if (scores[j] > max_val) max_val = scores[j];
            }
            float sum = 0;
            for (int j = 0; j < T; j++) sum += expf(scores[j] - max_val);
            for (int j = 0; j < T; j++)
                ref[row*T+j] = fp32_to_fp16(expf(scores[j] - max_val) / sum);
        }
        return compare_fp16(output, ref.data(), content_sz, 3, "softmax");
    }};
}

// =========================================================================
// 21. dual_argmax_kernel
// =========================================================================

static TestSetup setup_dual_argmax(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("dual_argmax_kernel");
    if (!k) return {"dual_argmax_kernel", 0, nullptr};

    const int vocab = 1024, total = 1280;
    auto bl = gpu.gpu_malloc(total*2), bo = gpu.gpu_malloc(2*4);
    auto* logits = (uint16_t*)bl.cpu_ptr;
    auto* out = (int32_t*)bo.cpu_ptr;
    fill_fp16(logits, total, -2, 2);
    logits[42] = fp32_to_fp16(10.0f);
    logits[vocab + 7] = fp32_to_fp16(10.0f);
    out[0] = -1; out[1] = -1;
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t logits, out; int32_t vocab, total; } args =
        {bl.gpu_addr, bo.gpu_addr, vocab, total};
    launch_only(gpu, k, base, &args, sizeof(args), 1, 1, 1, 256, 1, 1);

    return {"dual_argmax_kernel", 2, [out]() {
        int32_t ref[2] = {42, 7};
        return compare_int(out, ref, 2, "argmax");
    }};
}

// =========================================================================
// 22. mel_normalize_kernel (float I/O)
// =========================================================================

static TestSetup setup_mel_normalize(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("mel_normalize_kernel");
    if (!k) return {"mel_normalize_kernel", 0, nullptr};

    const int n_frames = 64, n_valid = 48;
    int in_sz = n_frames * 128;
    int out_sz = 128 * n_valid;

    auto bi = gpu.gpu_malloc(in_sz * 4), bo = gpu.gpu_malloc(out_sz * 4);
    auto* mel_in = (float*)bi.cpu_ptr;
    auto* mel_out = (float*)bo.cpu_ptr;
    fill_fp32(mel_in, in_sz, -10, 10);
    memset(mel_out, 0, out_sz * 4);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t in_, out_; int32_t n_frames, n_valid; } args =
        {bi.gpu_addr, bo.gpu_addr, n_frames, n_valid};

    int block = ((n_valid + 31) / 32) * 32;
    if (block > 1024) block = 1024;
    if (block < 32) block = 32;
    launch_only(gpu, k, base, &args, sizeof(args), 128, 1, 1, block, 1, 1);

    return {"mel_normalize_kernel", out_sz, [mel_in, mel_out, n_frames, n_valid, out_sz]() {
        std::vector<float> ref(out_sz);
        for (int ch = 0; ch < 128; ch++) {
            float sum = 0;
            for (int f = 0; f < n_valid; f++) sum += mel_in[f*128+ch];
            float mean = sum / n_valid;
            float sq = 0;
            for (int f = 0; f < n_valid; f++) {
                float d = mel_in[f*128+ch] - mean;
                sq += d*d;
            }
            float std_val = sqrtf(sq / (n_valid > 1 ? n_valid - 1 : 1)) + 1e-5f;
            float inv_std = 1.0f / std_val;
            for (int f = 0; f < n_valid; f++)
                ref[ch*n_valid+f] = (mel_in[f*128+ch] - mean) * inv_std;
        }
        return compare_fp32(mel_out, ref.data(), out_sz, 1e-4f, 1e-6f, "mel_norm");
    }};
}

// =========================================================================
// 23. transpose_kernel (2D grid, block=(32,8), shared memory)
// =========================================================================

static TestSetup setup_transpose(GPU& gpu, CubinLoader& cubin, uint64_t base) {
    auto* k = cubin.find_kernel("transpose_kernel");
    if (!k) return {"transpose_kernel", 0, nullptr};

    const int M = 64, N = 128;
    int total = M * N;
    auto bx = gpu.gpu_malloc(total*2), by = gpu.gpu_malloc(total*2);
    auto* x = (uint16_t*)bx.cpu_ptr;
    auto* y = (uint16_t*)by.cpu_ptr;
    fill_fp16(x, total, -1, 1); memset(y, 0, total*2);
    __sync_synchronize();

    struct __attribute__((packed)) { uint64_t x, y; int32_t M, N; } args =
        {bx.gpu_addr, by.gpu_addr, M, N};
    launch_only(gpu, k, base, &args, sizeof(args), (N+31)/32, (M+31)/32, 1, 32, 8, 1);

    return {"transpose_kernel", total, [x, y, M, N, total]() {
        std::vector<uint16_t> ref(total);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                ref[j*M+i] = x[i*N+j];
        return compare_fp16(y, ref.data(), total, 0, "transpose");
    }};
}

// =========================================================================
// Main — batched single-pushbuffer execution
//
// Hardware limit: Blackwell allows at most 16 SEND_PCAS dispatches per
// pushbuffer. Beyond that, the extra dispatches are silently dropped.
// We batch into groups of ≤16, using a fresh GPU instance per batch
// (since the GPFIFO also stalls after processing any PB with SEND_PCAS).
// =========================================================================

int main() {
    fprintf(stderr, "=== test_kernels: all 23 kernels via cudaless ===\n\n");

    CubinLoader cubin;
    if (!cubin.load("kernels.cubin")) {
        fprintf(stderr, "FATAL: could not load kernels.cubin\n");
        return 1;
    }
    fprintf(stderr, "Loaded %zu kernels from CUBIN\n\n", cubin.kernels.size());

    srand(42);

    using SetupFn = TestSetup(*)(GPU&, CubinLoader&, uint64_t);
    SetupFn setups[] = {
        setup_add_relu,
        setup_cast_fp32_to_fp16,
        setup_silu_inplace,
        setup_residual_add,
        setup_glu,
        setup_bias_relu_nchw,
        setup_bias_add,
        setup_lstm_cell,
        setup_embed_concat,
        setup_concat_vectors,
        setup_reshape_chw_to_hcw,
        setup_transpose_0213,
        setup_depthwise_conv1d,
        setup_conv2d,
        setup_im2col,
        setup_pos_encoding,
        setup_split_transpose_qkv,
        setup_layer_norm,
        setup_residual_add_layer_norm,
        setup_fused_score_softmax,
        setup_dual_argmax,
        setup_mel_normalize,
        setup_transpose,
    };
    int n_tests = sizeof(setups) / sizeof(setups[0]);

    const int BATCH_SIZE = 16;  // max SEND_PCAS per pushbuffer on Blackwell
    int pass = 0, fail = 0, skip = 0;
    std::vector<TestSetup> all_tests;

    for (int batch_start = 0; batch_start < n_tests; batch_start += BATCH_SIZE) {
        int batch_end = batch_start + BATCH_SIZE;
        if (batch_end > n_tests) batch_end = n_tests;
        int batch_sz = batch_end - batch_start;

        fprintf(stderr, "--- Batch %d: kernels %d-%d (%d dispatches) ---\n",
                batch_start / BATCH_SIZE + 1, batch_start + 1, batch_end, batch_sz);

        GPU gpu;
        if (!gpu.init()) {
            fprintf(stderr, "FATAL: GPU init failed for batch\n");
            return 1;
        }
        uint64_t cubin_base = gpu.upload_cubin(cubin.image.data(), cubin.image.size());

        // Setup + launch (all appended to one pushbuffer)
        gpu.begin_commands();
        std::vector<TestSetup> batch_tests;
        for (int i = batch_start; i < batch_end; i++)
            batch_tests.push_back(setups[i](gpu, cubin, cubin_base));

        // Submit + wait
        gpu.wait_kernel();
        usleep(5000);
        __sync_synchronize();

        // Verify this batch
        for (auto& t : batch_tests) {
            if (!t.verify) {
                fprintf(stderr, "[SKIP] %-40s  (not found in CUBIN)\n", t.name);
                skip++;
            } else {
                int mis = t.verify();
                if (mis == 0) {
                    fprintf(stderr, "[PASS] %-40s  %d/%d\n", t.name, t.total, t.total);
                    pass++;
                } else {
                    fprintf(stderr, "[FAIL] %-40s  %d/%d mismatches\n", t.name, mis, t.total);
                    fail++;
                }
            }
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "%d/%d PASS", pass, n_tests);
    if (fail) fprintf(stderr, ", %d FAIL", fail);
    if (skip) fprintf(stderr, ", %d SKIP", skip);
    fprintf(stderr, "\n");

    return fail ? 1 : 0;
}
