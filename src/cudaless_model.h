// cudaless_model.h — Full inference model using direct GPU ioctls (no libcuda/libcudart)
//
// Single-header: weight loading, buffer management, kernel wrappers,
// encoder (conformer), decoder (TDT), mel spectrogram.
//
// All GPU pointers are uint64_t (GPU virtual addresses). CPU-accessible via
// GpuAlloc.cpu_ptr from gpu.h's gpu_malloc().

#pragma once

#include "gpu.h"
#include "cubin_loader.h"
#include "cutlass_cudaless.h"

// MelFBEntry: matches struct in kernels.h (no CUDA headers needed)
struct CL_MelFBEntry {
    uint16_t freq;
    uint16_t mel;
    float weight;
};

// Mel data arrays are defined in mel_data.h (CUDA-free subset of mel.h)
#include "mel_data.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// =========================================================================
// Constants (from common.h and conformer.h, no CUDA deps)
// =========================================================================

static constexpr int CL_N_FFT = 512;
static constexpr int CL_HOP = 160;
static constexpr int CL_N_MELS = 128;
static constexpr int CL_N_FREQ = CL_N_FFT / 2 + 1;  // 257
static constexpr float CL_PREEMPH = 0.97f;
static constexpr float CL_LOG_EPS = 5.9604645e-08f;
static constexpr float CL_NORM_EPS = 1e-05f;

static constexpr int CL_D_MODEL    = 1024;
static constexpr int CL_D_FF       = 4096;
static constexpr int CL_N_HEADS    = 8;
static constexpr int CL_HEAD_DIM   = CL_D_MODEL / CL_N_HEADS;  // 128
static constexpr int CL_N_BLOCKS   = 24;
static constexpr int CL_D_CONV_PW  = 2048;
static constexpr int CL_CONV_K     = 9;
static constexpr int CL_SUB_CHANNELS = 256;
static constexpr int CL_D_PRED     = 640;
static constexpr int CL_N_VOCAB    = 1025;
static constexpr int CL_D_JOINT    = 640;
static constexpr int CL_D_OUTPUT   = 1030;

// =========================================================================
// FP16 helpers (no CUDA headers)
// =========================================================================

static inline uint16_t cl_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127;
    uint32_t frac = x & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) return sign;
    return sign | ((exp + 15) << 10) | (frac >> 13);
}

static inline float cl_fp16_to_fp32(uint16_t h) {
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
// Helper: align up
// =========================================================================

static inline size_t cl_align_up(size_t x, size_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

// =========================================================================
// Weight file format constants (same as conformer.h)
// =========================================================================

static constexpr uint32_t CL_PRKT_MAGIC   = 0x544B5250;  // "PRKT"
static constexpr uint32_t CL_PRKT_VERSION = 1;
static constexpr size_t   CL_HEADER_ALIGN = 4096;

// =========================================================================
// TensorDesc (from conformer.cpp)
// =========================================================================

struct CL_TensorDesc {
    std::string name;
    size_t offset;
    size_t size_bytes;
    std::string dtype;
    std::vector<int> shape;
};

// =========================================================================
// CudalessWeights — all model weights as GPU virtual addresses
// =========================================================================

struct CudalessWeights {
    GPU::GpuAlloc gpu_alloc = {};  // single contiguous allocation
    size_t gpu_data_size = 0;

    // Prefetch state
    void* mmap_ptr = nullptr;
    size_t mmap_size = 0;
    size_t data_offset = 0;

    std::vector<CL_TensorDesc> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;

    // Subsampling
    struct SubConv { uint64_t weight = 0, bias = 0; };
    SubConv sub_conv[7];
    uint64_t sub_out_w = 0, sub_out_b = 0;

    // Conformer blocks (x24)
    struct ConformerBlock {
        uint64_t ff1_ln_w = 0, ff1_ln_b = 0;
        uint64_t ff1_w1 = 0, ff1_w2 = 0;
        uint64_t mhsa_ln_w = 0, mhsa_ln_b = 0;
        uint64_t q_w = 0, k_w = 0, v_w = 0;
        uint64_t pos_w = 0, pos_bias_u = 0, pos_bias_v = 0;
        uint64_t out_w = 0;
        uint64_t conv_ln_w = 0, conv_ln_b = 0;
        uint64_t conv_pw1_w = 0, conv_dw_w = 0, conv_dw_b = 0, conv_pw2_w = 0;
        uint64_t ff2_ln_w = 0, ff2_ln_b = 0;
        uint64_t ff2_w1 = 0, ff2_w2 = 0;
        uint64_t final_ln_w = 0, final_ln_b = 0;
    } blocks[24];

    // Decoder
    uint64_t embed_w = 0;
    uint64_t lstm0_w_ih = 0, lstm0_w_hh = 0, lstm0_bias = 0;
    uint64_t lstm1_w_ih = 0, lstm1_w_hh = 0, lstm1_bias = 0;

    // Joint
    uint64_t enc_proj_w = 0, enc_proj_b = 0;
    uint64_t dec_proj_w = 0, dec_proj_b = 0;
    uint64_t out_proj_w = 0, out_proj_b = 0;

    // -----------------------------------------------------------------------
    // prefetch — CPU only (mmap + parse header)
    // -----------------------------------------------------------------------
    static CudalessWeights prefetch(const std::string& path) {
        CudalessWeights w;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "Cannot open weights: %s\n", path.c_str()); std::exit(1); }
        struct stat st;
        fstat(fd, &st);
        size_t file_size = st.st_size;
        void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        close(fd);
        if (mapped == MAP_FAILED) { fprintf(stderr, "mmap failed: %s\n", path.c_str()); std::exit(1); }
        madvise(mapped, file_size, MADV_SEQUENTIAL);
        const uint8_t* base = (const uint8_t*)mapped;

        uint32_t magic; memcpy(&magic, base, 4);
        if (magic != CL_PRKT_MAGIC) { fprintf(stderr, "Bad magic\n"); std::exit(1); }
        uint32_t version; memcpy(&version, base + 4, 4);
        if (version != CL_PRKT_VERSION) { fprintf(stderr, "Bad version %u\n", version); std::exit(1); }

        uint64_t header_len; memcpy(&header_len, base + 8, 8);
        // Parse header text
        std::string text((const char*)(base + 16), header_len);
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            std::istringstream ls(line);
            CL_TensorDesc td;
            ls >> td.name >> td.offset >> td.size_bytes >> td.dtype;
            int d; while (ls >> d) td.shape.push_back(d);
            w.tensors.push_back(std::move(td));
        }
        for (size_t i = 0; i < w.tensors.size(); i++)
            w.name_to_idx[w.tensors[i].name] = i;

        size_t header_end = 16 + header_len;
        size_t data_start = cl_align_up(header_end, CL_HEADER_ALIGN);

        size_t total_data = 0;
        for (auto& td : w.tensors) {
            size_t end = td.offset + td.size_bytes;
            if (end > total_data) total_data = end;
        }
        if (!w.tensors.empty()) {
            auto& last = w.tensors.back();
            total_data = std::max(total_data, cl_align_up(last.offset + last.size_bytes, 256));
        }

        w.gpu_data_size = total_data;
        w.mmap_ptr = mapped;
        w.mmap_size = file_size;
        w.data_offset = data_start;
        return w;
    }

    // -----------------------------------------------------------------------
    // upload — gpu_malloc + memcpy + assign pointers
    // -----------------------------------------------------------------------
    void upload(GPU& gpu) {
        const uint8_t* base = (const uint8_t*)mmap_ptr;
        gpu_alloc = gpu.gpu_malloc(gpu_data_size);
        memcpy(gpu_alloc.cpu_ptr, base + data_offset, gpu_data_size);
        __sync_synchronize();

        munmap(mmap_ptr, mmap_size);
        mmap_ptr = nullptr;
        mmap_size = 0;
        data_offset = 0;

        assign_pointers();
    }

    void assign_pointers() {
        uint64_t gpu_base = gpu_alloc.gpu_addr;
        auto lookup = [&](const std::string& name) -> uint64_t {
            auto it = name_to_idx.find(name);
            if (it == name_to_idx.end()) return 0;
            return gpu_base + tensors[it->second].offset;
        };

        for (int i : {0, 2, 3, 5, 6}) {
            std::string pre = "encoder/pre_encode.conv." + std::to_string(i);
            sub_conv[i].weight = lookup(pre + ".weight");
            sub_conv[i].bias   = lookup(pre + ".bias");
        }
        sub_out_w = lookup("encoder/pre_encode.out.weight");
        sub_out_b = lookup("encoder/pre_encode.out.bias");

        for (int i = 0; i < 24; i++) {
            auto& blk = blocks[i];
            std::string pre = "encoder/layers." + std::to_string(i);
            blk.ff1_ln_w = lookup(pre + ".norm_feed_forward1.weight");
            blk.ff1_ln_b = lookup(pre + ".norm_feed_forward1.bias");
            blk.ff1_w1   = lookup(pre + ".feed_forward1.linear1.weight");
            blk.ff1_w2   = lookup(pre + ".feed_forward1.linear2.weight");
            blk.mhsa_ln_w = lookup(pre + ".norm_self_att.weight");
            blk.mhsa_ln_b = lookup(pre + ".norm_self_att.bias");
            blk.q_w       = lookup(pre + ".self_attn.linear_q.weight");
            blk.k_w       = lookup(pre + ".self_attn.linear_k.weight");
            blk.v_w       = lookup(pre + ".self_attn.linear_v.weight");
            blk.pos_w     = lookup(pre + ".self_attn.linear_pos.weight");
            blk.pos_bias_u = lookup(pre + ".self_attn.pos_bias_u");
            blk.pos_bias_v = lookup(pre + ".self_attn.pos_bias_v");
            blk.out_w     = lookup(pre + ".self_attn.linear_out.weight");
            blk.conv_ln_w  = lookup(pre + ".norm_conv.weight");
            blk.conv_ln_b  = lookup(pre + ".norm_conv.bias");
            blk.conv_pw1_w = lookup(pre + ".conv.pointwise_conv1.weight");
            blk.conv_dw_w  = lookup(pre + ".conv.depthwise_conv.weight");
            blk.conv_dw_b  = lookup(pre + ".conv.depthwise_conv.bias");
            blk.conv_pw2_w = lookup(pre + ".conv.pointwise_conv2.weight");
            blk.ff2_ln_w = lookup(pre + ".norm_feed_forward2.weight");
            blk.ff2_ln_b = lookup(pre + ".norm_feed_forward2.bias");
            blk.ff2_w1   = lookup(pre + ".feed_forward2.linear1.weight");
            blk.ff2_w2   = lookup(pre + ".feed_forward2.linear2.weight");
            blk.final_ln_w = lookup(pre + ".norm_out.weight");
            blk.final_ln_b = lookup(pre + ".norm_out.bias");
        }

        embed_w = lookup("decoder/decoder.prediction.embed.weight");
        lstm0_w_ih = lookup("decoder/decoder.dec_rnn.lstm.weight_ih");
        lstm0_w_hh = lookup("decoder/decoder.dec_rnn.lstm.weight_hh");
        lstm0_bias = lookup("decoder/decoder.dec_rnn.lstm.bias");
        lstm1_w_ih = lookup("decoder/decoder.dec_rnn.lstm.1.weight_ih");
        lstm1_w_hh = lookup("decoder/decoder.dec_rnn.lstm.1.weight_hh");
        lstm1_bias = lookup("decoder/decoder.dec_rnn.lstm.1.bias");

        enc_proj_w = lookup("decoder/joint.enc.weight");
        enc_proj_b = lookup("decoder/joint.enc.bias");
        dec_proj_w = lookup("decoder/joint.pred.weight");
        dec_proj_b = lookup("decoder/joint.pred.bias");
        out_proj_w = lookup("decoder/joint.joint_net.joint_net.2.weight");
        out_proj_b = lookup("decoder/joint.joint_net.2.bias");
    }

    // Get CPU pointer for a weight (for CPU-side manipulation during init)
    void* cpu_ptr_for(uint64_t gpu_addr) const {
        if (gpu_addr == 0) return nullptr;
        uint64_t off = gpu_addr - gpu_alloc.gpu_addr;
        return (uint8_t*)gpu_alloc.cpu_ptr + off;
    }
};

// =========================================================================
// KernelLauncher — wrapper methods for all 24 custom kernels
// =========================================================================

struct KernelLauncher {
    GPU* gpu = nullptr;
    CubinLoader cubin;
    uint64_t cubin_base = 0;

    // Pre-looked-up kernel pointers
    CubinKernel* k_add_relu = nullptr;
    CubinKernel* k_bias_add = nullptr;
    CubinKernel* k_bias_relu_nchw = nullptr;
    CubinKernel* k_cast_fp32_to_fp16 = nullptr;
    CubinKernel* k_concat_vectors = nullptr;
    CubinKernel* k_conv2d = nullptr;
    CubinKernel* k_depthwise_conv1d_k9_silu = nullptr;
    CubinKernel* k_dual_argmax = nullptr;
    CubinKernel* k_embed_concat = nullptr;
    CubinKernel* k_fft512_mel_log = nullptr;
    CubinKernel* k_fused_score_softmax = nullptr;
    CubinKernel* k_glu = nullptr;
    CubinKernel* k_generate_pos_encoding = nullptr;
    CubinKernel* k_im2col = nullptr;
    CubinKernel* k_layer_norm = nullptr;
    CubinKernel* k_lstm_cell = nullptr;
    CubinKernel* k_mel_normalize = nullptr;
    CubinKernel* k_reshape_chw_to_hcw = nullptr;
    CubinKernel* k_residual_add = nullptr;
    CubinKernel* k_residual_add_layer_norm = nullptr;
    CubinKernel* k_silu_inplace = nullptr;
    CubinKernel* k_split_transpose_qkv_bias = nullptr;
    CubinKernel* k_transpose = nullptr;
    CubinKernel* k_transpose_0213 = nullptr;

    bool init(GPU& g, const char* cubin_path) {
        gpu = &g;
        if (!cubin.load(cubin_path)) {
            fprintf(stderr, "KernelLauncher: failed to load %s\n", cubin_path);
            return false;
        }
        cubin_base = g.upload_cubin(cubin.image.data(), cubin.image.size());
        if (!cubin_base) return false;
        fprintf(stderr, "KernelLauncher: cubin_base=0x%lx size=%zu\n",
                (unsigned long)cubin_base, cubin.image.size());

        k_add_relu               = cubin.find_kernel("add_relu_kernel");
        k_bias_add               = cubin.find_kernel("bias_add_kernel");
        k_bias_relu_nchw         = cubin.find_kernel("bias_relu_nchw_kernel");
        k_cast_fp32_to_fp16      = cubin.find_kernel("cast_fp32_to_fp16_kernel");
        k_concat_vectors         = cubin.find_kernel("concat_vectors_kernel");
        k_conv2d                 = cubin.find_kernel("conv2d_kernel");
        k_depthwise_conv1d_k9_silu = cubin.find_kernel("depthwise_conv1d_k9_silu_kernel");
        k_dual_argmax            = cubin.find_kernel("dual_argmax_kernel");
        k_embed_concat           = cubin.find_kernel("embed_concat_kernel");
        k_fft512_mel_log         = cubin.find_kernel("fft512_mel_log_kernel");
        k_fused_score_softmax    = cubin.find_kernel("fused_score_softmax_kernel");
        k_glu                    = cubin.find_kernel("glu_kernel");
        k_generate_pos_encoding  = cubin.find_kernel("generate_pos_encoding_kernel");
        k_im2col                 = cubin.find_kernel("im2col_2d_kernel");
        k_layer_norm             = cubin.find_kernel("layer_norm_kernel");
        k_lstm_cell              = cubin.find_kernel("lstm_cell_kernel");
        k_mel_normalize          = cubin.find_kernel("mel_normalize_kernel");
        k_reshape_chw_to_hcw    = cubin.find_kernel("reshape_chw_to_hcw_kernel");
        k_residual_add           = cubin.find_kernel("residual_add_kernel");
        k_residual_add_layer_norm = cubin.find_kernel("residual_add_layer_norm_kernel");
        k_silu_inplace           = cubin.find_kernel("silu_inplace_kernel");
        k_split_transpose_qkv_bias = cubin.find_kernel("split_transpose_qkv_bias_kernel");
        k_transpose              = cubin.find_kernel("transpose_kernel");
        k_transpose_0213         = cubin.find_kernel("transpose_0213_kernel");
        return true;
    }

    // Helper: launch kernel (optional cbuf3 for __constant__ data)
    void launch(CubinKernel* k, const void* args, uint32_t args_size,
                uint32_t gx, uint32_t gy, uint32_t gz,
                uint32_t bx, uint32_t by, uint32_t bz,
                uint64_t cbuf3_gpu = 0, uint32_t cbuf3_size = 0) {
        auto cbuf = gpu->prepare_cbuf0(args, args_size, k->cbuf0_size, k->param_base,
                                       bx, by, bz);
        gpu->launch_kernel(cubin_base + k->code_offset, k->code_size,
                           k->reg_count, k->shared_mem_size,
                           gx, gy, gz, bx, by, bz,
                           cbuf.gpu_addr, k->cbuf0_size,
                           cbuf3_gpu, cbuf3_size);
    }

    // --- Kernel wrapper methods ---
    // Each matches the arg struct from test_kernels.cpp exactly

    void add_relu(uint64_t a, uint64_t b, uint64_t y, int N) {
        struct __attribute__((packed)) { uint64_t a, b, y; int32_t n; } args = {a, b, y, N};
        launch(k_add_relu, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);
    }

    void cast_fp32_to_fp16(uint64_t x, uint64_t y, int N) {
        struct __attribute__((packed)) { uint64_t x, y; int32_t n; } args = {x, y, N};
        launch(k_cast_fp32_to_fp16, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);
    }

    void silu_inplace(uint64_t x, int N) {
        struct __attribute__((packed)) { uint64_t x; int32_t n; } args = {x, N};
        launch(k_silu_inplace, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);
    }

    void residual_add(uint64_t a, uint64_t b, uint64_t y, int N, float alpha) {
        struct __attribute__((packed)) { uint64_t a, b, y; int32_t n; float alpha; } args = {a, b, y, N, alpha};
        launch(k_residual_add, &args, sizeof(args), (N+255)/256, 1, 1, 256, 1, 1);
    }

    void glu(uint64_t x, uint64_t y, int N, int D) {
        int total = N * D;
        struct __attribute__((packed)) { uint64_t x, y; int32_t N, D; } args = {x, y, N, D};
        launch(k_glu, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);
    }

    void bias_relu_nchw(uint64_t x, uint64_t bias, int C, int spatial) {
        int total = C * spatial;
        struct __attribute__((packed)) { uint64_t x, bias; int32_t C, spatial; } args = {x, bias, C, spatial};
        launch(k_bias_relu_nchw, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);
    }

    void bias_add(uint64_t x, uint64_t bias, int rows, int cols) {
        int total = rows * cols;
        struct __attribute__((packed)) { uint64_t x, bias; int32_t rows, cols; } args = {x, bias, rows, cols};
        launch(k_bias_add, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);
    }

    void lstm_cell(uint64_t gates, uint64_t c_prev, uint64_t h_out, uint64_t c_out, int H) {
        struct __attribute__((packed)) { uint64_t gates, c_prev, h_out, c_out; int32_t H; } args =
            {gates, c_prev, h_out, c_out, H};
        launch(k_lstm_cell, &args, sizeof(args), (H+255)/256, 1, 1, 256, 1, 1);
    }

    void embed_concat(uint64_t table, int idx, uint64_t h, uint64_t out, int D) {
        struct __attribute__((packed)) {
            uint64_t table; int32_t idx; int32_t _pad;
            uint64_t h, out; int32_t D;
        } args = {table, idx, 0, h, out, D};
        launch(k_embed_concat, &args, sizeof(args), 1, 1, 1, 256, 1, 1);
    }

    void concat_vectors(uint64_t a, uint64_t b, uint64_t out, int D) {
        struct __attribute__((packed)) { uint64_t a, b, out; int32_t D; } args = {a, b, out, D};
        launch(k_concat_vectors, &args, sizeof(args), 1, 1, 1, 256, 1, 1);
    }

    void reshape_chw_to_hcw(uint64_t in, uint64_t out, int C, int H, int W) {
        int total = C * H * W;
        struct __attribute__((packed)) { uint64_t in_, out_; int32_t C, H, W; } args = {in, out, C, H, W};
        launch(k_reshape_chw_to_hcw, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);
    }

    void transpose_0213(uint64_t in, uint64_t out, int A, int B, int C) {
        int total = A * B * C;
        struct __attribute__((packed)) { uint64_t in_, out_; int32_t A, B, C; } args = {in, out, A, B, C};
        launch(k_transpose_0213, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);
    }

    void depthwise_conv1d_k9_silu(uint64_t x, uint64_t w, uint64_t b, uint64_t y, int T, int C) {
        int total = T * C;
        struct __attribute__((packed)) { uint64_t x, w, b, y; int32_t T, C; } args = {x, w, b, y, T, C};
        launch(k_depthwise_conv1d_k9_silu, &args, sizeof(args), (total+255)/256, 1, 1, 256, 1, 1);
    }

    void conv2d(uint64_t input, uint64_t weight, uint64_t bias, uint64_t output,
                int C_in, int H_in, int W_in, int C_out, int kH, int kW,
                int stride, int pad, int groups) {
        int H_out = (H_in + 2*pad - kH) / stride + 1;
        int W_out = (W_in + 2*pad - kW) / stride + 1;
        int c_per_group = C_in / groups;
        int out_sz = C_out * H_out * W_out;
        struct __attribute__((packed)) {
            uint64_t input, weight, bias, output;
            int32_t C_in, H_in, W_in, C_out, H_out, W_out;
            int32_t kH, kW, stride, pad, groups, c_per_group;
        } args = {input, weight, bias, output,
                  C_in, H_in, W_in, C_out, H_out, W_out, kH, kW, stride, pad, groups, c_per_group};
        launch(k_conv2d, &args, sizeof(args), (out_sz+255)/256, 1, 1, 256, 1, 1);
    }

    void generate_pos_encoding(uint64_t output, int T, int d_model) {
        int half_d = d_model / 2;
        int total_work = (2*T - 1) * half_d;
        struct __attribute__((packed)) { uint64_t output; int32_t T, d_model; } args = {output, T, d_model};
        launch(k_generate_pos_encoding, &args, sizeof(args), (total_work+255)/256, 1, 1, 256, 1, 1);
    }

    void split_transpose_qkv_bias(uint64_t in, uint64_t bu, uint64_t bv,
                                   uint64_t qu, uint64_t qv, uint64_t k, uint64_t v,
                                   int T, int heads, int hd) {
        int total_work = T * heads * hd;
        struct __attribute__((packed)) {
            uint64_t in_, bu_, bv_, qu_, qv_, k_, v_;
            int32_t T, heads, hd;
        } args = {in, bu, bv, qu, qv, k, v, T, heads, hd};
        launch(k_split_transpose_qkv_bias, &args, sizeof(args), (total_work+255)/256, 1, 1, 256, 1, 1);
    }

    void layer_norm(uint64_t x, uint64_t gamma, uint64_t beta, uint64_t y,
                    int N, int D, float eps) {
        struct __attribute__((packed)) {
            uint64_t x, gamma, beta, y; int32_t N, D; float eps;
        } args = {x, gamma, beta, y, N, D, eps};
        int block = ((D + 31) / 32) * 32;
        if (block > 1024) block = 1024;
        if (block < 32) block = 32;
        launch(k_layer_norm, &args, sizeof(args), N, 1, 1, block, 1, 1);
    }

    void residual_add_layer_norm(uint64_t x, uint64_t delta, float alpha,
                                  uint64_t gamma, uint64_t beta, uint64_t ln_out,
                                  int N, int D, float eps) {
        struct __attribute__((packed)) {
            uint64_t x, delta; float alpha; int32_t _pad;
            uint64_t gamma, beta, ln_out;
            int32_t N, D; float eps;
        } args = {x, delta, alpha, 0, gamma, beta, ln_out, N, D, eps};
        int block = ((D + 31) / 32) * 32;
        if (block > 1024) block = 1024;
        if (block < 32) block = 32;
        launch(k_residual_add_layer_norm, &args, sizeof(args), N, 1, 1, block, 1, 1);
    }

    void fused_score_softmax(uint64_t content, uint64_t pos, uint64_t output,
                              int heads, int T, float scale) {
        int rows = heads * T;
        struct __attribute__((packed)) {
            uint64_t content, pos, output; int32_t T; float scale;
        } args = {content, pos, output, T, scale};
        int block = ((T + 31) / 32) * 32;
        if (block > 1024) block = 1024;
        if (block < 32) block = 32;
        launch(k_fused_score_softmax, &args, sizeof(args), rows, 1, 1, block, 1, 1);
    }

    void dual_argmax(uint64_t logits, uint64_t out, int vocab, int total) {
        struct __attribute__((packed)) { uint64_t logits, out; int32_t vocab, total; } args =
            {logits, out, vocab, total};
        launch(k_dual_argmax, &args, sizeof(args), 1, 1, 1, 256, 1, 1);
    }

    void mel_normalize(uint64_t in, uint64_t out, int n_frames, int n_valid) {
        struct __attribute__((packed)) { uint64_t in_, out_; int32_t n_frames, n_valid; } args =
            {in, out, n_frames, n_valid};
        int block = ((n_valid + 31) / 32) * 32;
        if (block > 1024) block = 1024;
        if (block < 32) block = 32;
        launch(k_mel_normalize, &args, sizeof(args), 128, 1, 1, block, 1, 1);
    }

    void transpose_2d(uint64_t x, uint64_t y, int M, int N) {
        struct __attribute__((packed)) { uint64_t x, y; int32_t M, N; } args = {x, y, M, N};
        launch(k_transpose, &args, sizeof(args), (N+31)/32, (M+31)/32, 1, 32, 8, 1);
    }

    void fft512_mel_log(uint64_t frames, uint64_t mel_out, int n_frames,
                        uint64_t cbuf3_gpu, uint32_t cbuf3_size) {
        struct __attribute__((packed)) { uint64_t frames, mel_out; int32_t n_frames; } args =
            {frames, mel_out, n_frames};
        launch(k_fft512_mel_log, &args, sizeof(args), n_frames, 1, 1, 256, 1, 1,
               cbuf3_gpu, cbuf3_size);
    }
};

// =========================================================================
// Buf — GPU address + CPU pointer pair
// =========================================================================

struct Buf {
    uint64_t gpu;
    void* cpu;
};

// =========================================================================
// CudalessModel — full inference model
// =========================================================================

struct CudalessModel {
    GPU* gpu = nullptr;
    const CudalessWeights* w = nullptr;
    KernelLauncher kl;
    CutlassCudaless cutlass;

    int T_max = 0;

    // Pre-concatenated QKV weights per block [D_MODEL, 3*D_MODEL]
    Buf qkv_w[CL_N_BLOCKS];

    // Pre-combined LSTM weights [4*D_PRED, 2*D_PRED] per layer
    Buf lstm_combined_w[2];
    Buf lstm_combined_bias[2];  // [4*D_PRED]

    // Encoder buffers
    Buf sub_buf[2];
    Buf mel_fp16;
    Buf mel_fp32_buf;
    Buf x, ln_out, ff_mid, ff_out;
    Buf qkv, q, k, v;
    Buf pos_enc, pos_proj;
    Buf q_u, q_v_buf;
    Buf scores, pos_scores;
    Buf attn_out, mhsa_out;
    Buf conv_mid, conv_glu, conv_dw;

    // Decoder buffers
    Buf lstm_input, lstm_gates;
    Buf lstm_h[2], lstm_c[2];
    Buf lstm_h_out[2], lstm_c_out[2];
    Buf enc_proj_all;
    Buf dec_proj_buf, joint_act, joint_out;
    Buf argmax_out;

    // Mel filterbank on GPU (for cbuf3 — SM120 __constant__ data)
    GPU::GpuAlloc mel_fb_gpu = {};

    // Mel frame upload buffer
    GPU::GpuAlloc mel_frames_gpu = {};
    GPU::GpuAlloc mel_raw_gpu = {};
    int mel_frames_cap = 0;

    // -----------------------------------------------------------------------
    // init
    // -----------------------------------------------------------------------

    bool init(GPU& g, const CudalessWeights& weights, int max_mel_frames) {
        gpu = &g;
        w = &weights;
        T_max = max_mel_frames / 8 + 10;

        // Init kernel launcher
        if (!kl.init(g, "kernels.cubin")) return false;

        // Init CUTLASS
        if (!cutlass.init(g, "cutlass_gemm.cubin")) return false;

        // Upload mel filterbank to GPU (for fft512_mel_log cbuf3)
        uint32_t cbuf3_sz = kl.cubin.cbuf3_size;
        if (cbuf3_sz < N_MEL_ENTRIES * (uint32_t)sizeof(CL_MelFBEntry))
            cbuf3_sz = N_MEL_ENTRIES * sizeof(CL_MelFBEntry);
        mel_fb_gpu = g.gpu_malloc(cbuf3_sz);
        memset(mel_fb_gpu.cpu_ptr, 0, cbuf3_sz);
        memcpy(mel_fb_gpu.cpu_ptr, MEL_FILTERBANK, N_MEL_ENTRIES * sizeof(CL_MelFBEntry));
        __sync_synchronize();

        // Allocate all inference buffers from single pool
        int max_sub = CL_SUB_CHANNELS * (max_mel_frames / 2 + 1) * 65;
        size_t mel_fp32_elems = 128 * max_mel_frames;

        size_t sizes[] = {
            (size_t)max_sub,                              // sub_buf[0]
            (size_t)max_sub,                              // sub_buf[1]
            (size_t)(128 * max_mel_frames),               // mel_fp16
            (size_t)(T_max * CL_D_MODEL),                 // x
            (size_t)(T_max * CL_D_MODEL),                 // ln_out
            (size_t)(T_max * CL_D_FF),                    // ff_mid
            (size_t)(T_max * CL_D_MODEL),                 // ff_out
            (size_t)(T_max * 3 * CL_D_MODEL),             // qkv
            (size_t)(CL_N_HEADS * T_max * CL_HEAD_DIM),   // q
            (size_t)(CL_N_HEADS * T_max * CL_HEAD_DIM),   // k
            (size_t)(CL_N_HEADS * T_max * CL_HEAD_DIM),   // v
            (size_t)((2 * T_max) * CL_D_MODEL),           // pos_enc
            (size_t)((2 * T_max) * CL_D_MODEL),           // pos_proj
            (size_t)(CL_N_HEADS * T_max * CL_HEAD_DIM),   // q_u
            (size_t)(CL_N_HEADS * T_max * CL_HEAD_DIM),   // q_v_buf
            (size_t)(CL_N_HEADS * T_max * T_max),         // scores
            (size_t)(CL_N_HEADS * T_max * (2*T_max)),     // pos_scores
            (size_t)(CL_N_HEADS * T_max * CL_HEAD_DIM),   // attn_out
            (size_t)(T_max * CL_D_MODEL),                 // mhsa_out
            (size_t)(T_max * CL_D_CONV_PW),               // conv_mid
            (size_t)(T_max * CL_D_MODEL),                 // conv_glu
            (size_t)(T_max * CL_D_MODEL),                 // conv_dw
            (size_t)(2 * CL_D_PRED),                      // lstm_input
            (size_t)(4 * CL_D_PRED),                      // lstm_gates
            (size_t)CL_D_PRED, (size_t)CL_D_PRED,         // lstm_h[0], lstm_h[1]
            (size_t)CL_D_PRED, (size_t)CL_D_PRED,         // lstm_c[0], lstm_c[1]
            (size_t)CL_D_PRED, (size_t)CL_D_PRED,         // lstm_h_out[0], lstm_h_out[1]
            (size_t)CL_D_PRED, (size_t)CL_D_PRED,         // lstm_c_out[0], lstm_c_out[1]
            (size_t)(T_max * CL_D_JOINT),                  // enc_proj_all
            (size_t)CL_D_JOINT, (size_t)CL_D_JOINT,       // dec_proj_buf, joint_act
            (size_t)CL_D_OUTPUT,                           // joint_out
            // Pre-combined LSTM weights
            (size_t)(4 * CL_D_PRED * 2 * CL_D_PRED),     // lstm_combined_w[0]
            (size_t)(4 * CL_D_PRED * 2 * CL_D_PRED),     // lstm_combined_w[1]
            (size_t)(4 * CL_D_PRED),                       // lstm_combined_bias[0]
            (size_t)(4 * CL_D_PRED),                       // lstm_combined_bias[1]
        };
        constexpr int N_BUFS = sizeof(sizes) / sizeof(sizes[0]);

        size_t total_half = 0;
        for (int i = 0; i < N_BUFS; i++) total_half += sizes[i];

        size_t qkv_total = (size_t)CL_N_BLOCKS * CL_D_MODEL * 3 * CL_D_MODEL;

        int total_bufs = N_BUFS + CL_N_BLOCKS + 2;
        size_t pool_bytes = (total_half + qkv_total) * sizeof(uint16_t)
                          + mel_fp32_elems * sizeof(float)
                          + 2 * sizeof(int)
                          + (size_t)total_bufs * 256;

        auto pool_alloc = g.gpu_malloc(pool_bytes);
        uint8_t* pool = (uint8_t*)pool_alloc.cpu_ptr;
        uint64_t pool_gpu = pool_alloc.gpu_addr;

        constexpr size_t ALIGN = 256;
        auto take = [&](size_t n_halfs) -> Buf {
            uintptr_t p = (uintptr_t)pool;
            p = (p + ALIGN - 1) & ~(ALIGN - 1);
            pool = (uint8_t*)p;
            uint64_t gpu_off = (uint64_t)(pool - (uint8_t*)pool_alloc.cpu_ptr);
            Buf b = { pool_gpu + gpu_off, pool };
            pool += n_halfs * sizeof(uint16_t);
            return b;
        };

        sub_buf[0] = take(sizes[0]);   sub_buf[1] = take(sizes[1]);
        mel_fp16   = take(sizes[2]);
        x          = take(sizes[3]);   ln_out     = take(sizes[4]);
        ff_mid     = take(sizes[5]);   ff_out     = take(sizes[6]);
        qkv        = take(sizes[7]);
        q          = take(sizes[8]);   k          = take(sizes[9]);
        v          = take(sizes[10]);
        pos_enc    = take(sizes[11]);  pos_proj   = take(sizes[12]);
        q_u        = take(sizes[13]);  q_v_buf    = take(sizes[14]);
        scores     = take(sizes[15]);  pos_scores = take(sizes[16]);
        attn_out   = take(sizes[17]);  mhsa_out   = take(sizes[18]);
        conv_mid   = take(sizes[19]);  conv_glu   = take(sizes[20]);
        conv_dw    = take(sizes[21]);
        lstm_input = take(sizes[22]);  lstm_gates = take(sizes[23]);
        lstm_h[0]  = take(sizes[24]);  lstm_h[1]  = take(sizes[25]);
        lstm_c[0]  = take(sizes[26]);  lstm_c[1]  = take(sizes[27]);
        lstm_h_out[0] = take(sizes[28]); lstm_h_out[1] = take(sizes[29]);
        lstm_c_out[0] = take(sizes[30]); lstm_c_out[1] = take(sizes[31]);
        enc_proj_all = take(sizes[32]);
        dec_proj_buf = take(sizes[33]); joint_act = take(sizes[34]);
        joint_out    = take(sizes[35]);
        lstm_combined_w[0]    = take(sizes[36]);
        lstm_combined_w[1]    = take(sizes[37]);
        lstm_combined_bias[0] = take(sizes[38]);
        lstm_combined_bias[1] = take(sizes[39]);

        for (int b = 0; b < CL_N_BLOCKS; b++)
            qkv_w[b] = take(CL_D_MODEL * 3 * CL_D_MODEL);

        // mel_fp32 (float*) from pool
        {
            uintptr_t p = (uintptr_t)pool;
            p = (p + ALIGN - 1) & ~(ALIGN - 1);
            pool = (uint8_t*)p;
            uint64_t gpu_off = (uint64_t)(pool - (uint8_t*)pool_alloc.cpu_ptr);
            mel_fp32_buf = { pool_gpu + gpu_off, pool };
            pool += mel_fp32_elems * sizeof(float);
        }

        // argmax_out (int[2]) from pool
        {
            uintptr_t p = (uintptr_t)pool;
            p = (p + ALIGN - 1) & ~(ALIGN - 1);
            pool = (uint8_t*)p;
            uint64_t gpu_off = (uint64_t)(pool - (uint8_t*)pool_alloc.cpu_ptr);
            argmax_out = { pool_gpu + gpu_off, pool };
            pool += 2 * sizeof(int);
        }

        // Pre-concatenate QKV weights: row-interleave Q,K,V into [D, 3D]
        // CPU-side memcpy since all GPU memory is CPU-accessible
        for (int b = 0; b < CL_N_BLOCKS; b++) {
            auto& blk = weights.blocks[b];
            const uint16_t* q_src = (const uint16_t*)weights.cpu_ptr_for(blk.q_w);
            const uint16_t* k_src = (const uint16_t*)weights.cpu_ptr_for(blk.k_w);
            const uint16_t* v_src = (const uint16_t*)weights.cpu_ptr_for(blk.v_w);
            uint16_t* dst = (uint16_t*)qkv_w[b].cpu;
            for (int row = 0; row < CL_D_MODEL; row++) {
                memcpy(dst + row * 3 * CL_D_MODEL, q_src + row * CL_D_MODEL, CL_D_MODEL * 2);
                memcpy(dst + row * 3 * CL_D_MODEL + CL_D_MODEL, k_src + row * CL_D_MODEL, CL_D_MODEL * 2);
                memcpy(dst + row * 3 * CL_D_MODEL + 2 * CL_D_MODEL, v_src + row * CL_D_MODEL, CL_D_MODEL * 2);
            }
        }

        // Pre-combine LSTM weights: [W_ih | W_hh] side by side, per layer
        {
            const uint64_t w_ih[2] = { weights.lstm0_w_ih, weights.lstm1_w_ih };
            const uint64_t w_hh[2] = { weights.lstm0_w_hh, weights.lstm1_w_hh };
            const uint64_t bias[2] = { weights.lstm0_bias, weights.lstm1_bias };
            for (int layer = 0; layer < 2; layer++) {
                const uint16_t* ih = (const uint16_t*)weights.cpu_ptr_for(w_ih[layer]);
                const uint16_t* hh = (const uint16_t*)weights.cpu_ptr_for(w_hh[layer]);
                uint16_t* dst = (uint16_t*)lstm_combined_w[layer].cpu;
                for (int row = 0; row < 4 * CL_D_PRED; row++) {
                    memcpy(dst + row * 2 * CL_D_PRED, ih + row * CL_D_PRED, CL_D_PRED * 2);
                    memcpy(dst + row * 2 * CL_D_PRED + CL_D_PRED, hh + row * CL_D_PRED, CL_D_PRED * 2);
                }

                // Pre-add biases: combined_bias = b_ih + b_hh
                const uint16_t* b_src = (const uint16_t*)weights.cpu_ptr_for(bias[layer]);
                uint16_t* b_dst = (uint16_t*)lstm_combined_bias[layer].cpu;
                for (int i = 0; i < 4 * CL_D_PRED; i++) {
                    float val = cl_fp16_to_fp32(b_src[i]) + cl_fp16_to_fp32(b_src[4 * CL_D_PRED + i]);
                    b_dst[i] = cl_fp32_to_fp16(val);
                }
            }
        }

        __sync_synchronize();

        // Pre-compute position encoding
        gpu->begin_commands();
        kl.generate_pos_encoding(pos_enc.gpu, T_max, CL_D_MODEL);
        gpu->wait_kernel();

        return true;
    }

    // -----------------------------------------------------------------------
    // GEMM convenience wrappers
    // -----------------------------------------------------------------------

    // Y[m,n] = X[m,k] @ W[k,n]
    void gnn(uint64_t X, int m, int k, uint64_t W, int n, uint64_t Y) {
        cutlass.gemm_nn(X, m, k, W, n, Y);
    }

    // Y[m,n] = X[m,k] @ W[n,k]^T
    void gnt(uint64_t X, int m, int k, uint64_t W, int n, uint64_t Y) {
        cutlass.gemm_nt(X, m, k, W, n, Y);
    }

    // Y[m,n] = X[m,k] @ W[k,n] + bias
    void gnn_bias(uint64_t X, int m, int k, uint64_t W, int n, uint64_t bias, uint64_t Y) {
        cutlass.gemm_nn(X, m, k, W, n, Y);
        kl.bias_add(Y, bias, m, n);
    }

    // Y[m,n] = X[m,k] @ W[n,k]^T + bias
    void gnt_bias(uint64_t X, int m, int k, uint64_t W, int n, uint64_t bias, uint64_t Y) {
        cutlass.gemm_nt(X, m, k, W, n, Y);
        kl.bias_add(Y, bias, m, n);
    }

    // -----------------------------------------------------------------------
    // Mel spectrogram (CPU preprocess + GPU FFT/normalize)
    // -----------------------------------------------------------------------

    void mel_compute(const float* audio, int num_samples,
                     int& n_frames, int& n_valid) {
        // CPU preemphasis + windowing
        std::vector<float> preemph(num_samples);
        preemph[0] = audio[0];
        for (int i = 1; i < num_samples; i++)
            preemph[i] = audio[i] - CL_PREEMPH * audio[i - 1];

        int pad = CL_N_FFT / 2;
        int padded_len = num_samples + 2 * pad;
        n_frames = (padded_len - CL_N_FFT) / CL_HOP + 1;
        n_valid = num_samples / CL_HOP;

        std::vector<float> frames(n_frames * CL_N_FFT, 0.0f);
        for (int f = 0; f < n_frames; f++) {
            float* row = frames.data() + f * CL_N_FFT;
            for (int i = 0; i < CL_N_FFT; i++) {
                int pos = f * CL_HOP + i - pad;
                float sample = (pos >= 0 && pos < num_samples) ? preemph[pos] : 0.0f;
                row[i] = sample * HANN_WINDOW[i];
            }
        }

        // Ensure GPU buffers
        size_t need_frames = (size_t)n_frames * CL_N_FFT * sizeof(float);
        size_t need_mel = (size_t)n_frames * CL_N_MELS * sizeof(float);
        if (n_frames > mel_frames_cap) {
            mel_frames_gpu = gpu->gpu_malloc(need_frames);
            mel_raw_gpu = gpu->gpu_malloc(need_mel);
            mel_frames_cap = n_frames;
        }

        // Upload windowed frames
        memcpy(mel_frames_gpu.cpu_ptr, frames.data(), need_frames);
        __sync_synchronize();

        // GPU: FFT + mel + log → normalize + transpose
        gpu->begin_commands();
        kl.fft512_mel_log(mel_frames_gpu.gpu_addr, mel_raw_gpu.gpu_addr, n_frames,
                          mel_fb_gpu.gpu_addr, mel_fb_gpu.size);
        kl.mel_normalize(mel_raw_gpu.gpu_addr, mel_fp32_buf.gpu, n_frames, n_valid);
        gpu->wait_kernel();
    }

    // -----------------------------------------------------------------------
    // encode_gpu — mel_fp32_buf already populated
    // -----------------------------------------------------------------------

    int encode_gpu(int T_mel) {
        gpu->begin_commands();

        // 1. Cast mel FP32→FP16, transpose [128,T] → [T,128]
        kl.cast_fp32_to_fp16(mel_fp32_buf.gpu, mel_fp16.gpu, 128 * T_mel);
        kl.transpose_2d(mel_fp16.gpu, sub_buf[0].gpu, 128, T_mel);

        // 2. Subsampling: [1, T_mel, 128] → [T', 1024]
        int H = T_mel, W = 128;
        int H2 = (H + 2*1 - 3) / 2 + 1;
        int W2 = (W + 2*1 - 3) / 2 + 1;  // 64
        kl.conv2d(sub_buf[0].gpu, w->sub_conv[0].weight, 0, sub_buf[1].gpu,
                  1, H, W, CL_SUB_CHANNELS, 3, 3, 2, 1, 1);
        kl.bias_relu_nchw(sub_buf[1].gpu, w->sub_conv[0].bias, CL_SUB_CHANNELS, H2 * W2);
        H = H2; W = W2;

        // conv.2 (depthwise) → conv.3 (pointwise) → bias+ReLU
        kl.conv2d(sub_buf[1].gpu, w->sub_conv[2].weight, w->sub_conv[2].bias, sub_buf[0].gpu,
                  CL_SUB_CHANNELS, H, W, CL_SUB_CHANNELS, 3, 3, 2, 1, CL_SUB_CHANNELS);
        H = (H + 2*1 - 3) / 2 + 1;
        W = (W + 2*1 - 3) / 2 + 1;  // 32
        gnn(w->sub_conv[3].weight, CL_SUB_CHANNELS, CL_SUB_CHANNELS, sub_buf[0].gpu, H * W, sub_buf[1].gpu);
        kl.bias_relu_nchw(sub_buf[1].gpu, w->sub_conv[3].bias, CL_SUB_CHANNELS, H * W);

        // conv.5 (depthwise) → conv.6 (pointwise) → bias+ReLU
        kl.conv2d(sub_buf[1].gpu, w->sub_conv[5].weight, w->sub_conv[5].bias, sub_buf[0].gpu,
                  CL_SUB_CHANNELS, H, W, CL_SUB_CHANNELS, 3, 3, 2, 1, CL_SUB_CHANNELS);
        H = (H + 2*1 - 3) / 2 + 1;
        W = (W + 2*1 - 3) / 2 + 1;  // 16
        gnn(w->sub_conv[6].weight, CL_SUB_CHANNELS, CL_SUB_CHANNELS, sub_buf[0].gpu, H * W, sub_buf[1].gpu);
        kl.bias_relu_nchw(sub_buf[1].gpu, w->sub_conv[6].bias, CL_SUB_CHANNELS, H * W);

        // Reshape [256, H, 16] → [H, 4096] then linear → [T', 1024]
        int T = H;
        kl.reshape_chw_to_hcw(sub_buf[1].gpu, sub_buf[0].gpu, CL_SUB_CHANNELS, T, W);
        if (w->sub_out_b)
            gnn_bias(sub_buf[0].gpu, T, CL_SUB_CHANNELS * W, w->sub_out_w, CL_D_MODEL, w->sub_out_b, x.gpu);
        else
            gnn(sub_buf[0].gpu, T, CL_SUB_CHANNELS * W, w->sub_out_w, CL_D_MODEL, x.gpu);

        // 3. Position encoding: slice from pre-computed T_max encoding
        uint64_t pos_enc_T = pos_enc.gpu + (size_t)(T_max - T) * CL_D_MODEL * sizeof(uint16_t);
        int pos_len = 2 * T - 1;

        // 4. Conformer blocks
        for (int blk = 0; blk < CL_N_BLOCKS; blk++) {
            gpu->wait_kernel();
            gpu->begin_commands();
            auto& b = w->blocks[blk];

            // --- FF1 (half-step residual) ---
            kl.layer_norm(x.gpu, b.ff1_ln_w, b.ff1_ln_b, ln_out.gpu, T, CL_D_MODEL, 1e-5f);
            gnn(ln_out.gpu, T, CL_D_MODEL, b.ff1_w1, CL_D_FF, ff_mid.gpu);
            kl.silu_inplace(ff_mid.gpu, T * CL_D_FF);
            gnn(ff_mid.gpu, T, CL_D_FF, b.ff1_w2, CL_D_MODEL, ff_out.gpu);
            kl.residual_add_layer_norm(x.gpu, ff_out.gpu, 0.5f,
                b.mhsa_ln_w, b.mhsa_ln_b, ln_out.gpu, T, CL_D_MODEL, 1e-5f);

            // --- MHSA ---
            {
                // Fused QKV projection
                gnn(ln_out.gpu, T, CL_D_MODEL, qkv_w[blk].gpu, 3 * CL_D_MODEL, qkv.gpu);

                // Fused split+transpose+bias
                kl.split_transpose_qkv_bias(qkv.gpu, b.pos_bias_u, b.pos_bias_v,
                    q_u.gpu, q_v_buf.gpu, k.gpu, v.gpu,
                    T, CL_N_HEADS, CL_HEAD_DIM);

                // Position encoding projection
                gnn(pos_enc_T, pos_len, CL_D_MODEL, b.pos_w, CL_D_MODEL, pos_proj.gpu);

                gpu->wait_kernel(); gpu->begin_commands(); // DEBUG
                // Position scores via strided batched GEMM
                cutlass.batched_gemm_nt_ex(
                    q_v_buf.gpu, CL_HEAD_DIM, (long long)T * CL_HEAD_DIM,
                    pos_proj.gpu, CL_D_MODEL, (long long)CL_HEAD_DIM,
                    pos_scores.gpu, pos_len, (long long)T * pos_len,
                    CL_N_HEADS, T, pos_len, CL_HEAD_DIM);
                gpu->wait_kernel(); gpu->begin_commands(); // DEBUG

                float scale = 1.0f / sqrtf((float)CL_HEAD_DIM);

                // Content attention scores
                cutlass.batched_gemm_nt(q_u.gpu, k.gpu, scores.gpu,
                    CL_N_HEADS, T, T, CL_HEAD_DIM,
                    (long long)T * CL_HEAD_DIM, (long long)T * CL_HEAD_DIM, (long long)T * T);
                gpu->wait_kernel(); gpu->begin_commands(); // DEBUG

                // Fused: (content + pos_skew) * scale + softmax
                kl.fused_score_softmax(scores.gpu, pos_scores.gpu, scores.gpu,
                    CL_N_HEADS, T, scale);
                gpu->wait_kernel(); gpu->begin_commands(); // DEBUG

                // Weighted sum
                cutlass.batched_gemm_nn(scores.gpu, v.gpu, attn_out.gpu,
                    CL_N_HEADS, T, CL_HEAD_DIM, T,
                    (long long)T * T, (long long)T * CL_HEAD_DIM, (long long)T * CL_HEAD_DIM);
                gpu->wait_kernel(); gpu->begin_commands(); // DEBUG

                // Transpose [8,T,128] → [T,8,128] = [T,1024]
                kl.transpose_0213(attn_out.gpu, ff_out.gpu, CL_N_HEADS, T, CL_HEAD_DIM);

                // Output projection
                gnn(ff_out.gpu, T, CL_D_MODEL, b.out_w, CL_D_MODEL, mhsa_out.gpu);
            }

            // Fused: x += mhsa_out, then LN
            kl.residual_add_layer_norm(x.gpu, mhsa_out.gpu, 1.0f,
                b.conv_ln_w, b.conv_ln_b, ln_out.gpu, T, CL_D_MODEL, 1e-5f);

            // Pointwise conv1 + GLU
            gnt(ln_out.gpu, T, CL_D_MODEL, b.conv_pw1_w, CL_D_CONV_PW, conv_mid.gpu);
            kl.glu(conv_mid.gpu, conv_glu.gpu, T, CL_D_MODEL);

            // Depthwise conv 1D k=9 + SiLU
            kl.depthwise_conv1d_k9_silu(conv_glu.gpu, b.conv_dw_w, b.conv_dw_b,
                                         conv_dw.gpu, T, CL_D_MODEL);

            // Pointwise conv2
            gnt(conv_dw.gpu, T, CL_D_MODEL, b.conv_pw2_w, CL_D_MODEL, mhsa_out.gpu);

            // Fused: x += conv_out, then LN
            kl.residual_add_layer_norm(x.gpu, mhsa_out.gpu, 1.0f,
                b.ff2_ln_w, b.ff2_ln_b, ln_out.gpu, T, CL_D_MODEL, 1e-5f);

            // FF2
            gnn(ln_out.gpu, T, CL_D_MODEL, b.ff2_w1, CL_D_FF, ff_mid.gpu);
            kl.silu_inplace(ff_mid.gpu, T * CL_D_FF);
            gnn(ff_mid.gpu, T, CL_D_FF, b.ff2_w2, CL_D_MODEL, ff_out.gpu);
            kl.residual_add_layer_norm(x.gpu, ff_out.gpu, 0.5f,
                b.final_ln_w, b.final_ln_b, x.gpu, T, CL_D_MODEL, 1e-5f);
        }

        // Encoder projection for all T frames
        gnn_bias(x.gpu, T, CL_D_MODEL, w->enc_proj_w, CL_D_JOINT, w->enc_proj_b, enc_proj_all.gpu);

        gpu->wait_kernel();
        return T;
    }

    // -----------------------------------------------------------------------
    // decoder_reset
    // -----------------------------------------------------------------------

    void decoder_reset() {
        memset(lstm_h[0].cpu, 0, CL_D_PRED * sizeof(uint16_t));
        memset(lstm_c[0].cpu, 0, CL_D_PRED * sizeof(uint16_t));
        memset(lstm_h[1].cpu, 0, CL_D_PRED * sizeof(uint16_t));
        memset(lstm_c[1].cpu, 0, CL_D_PRED * sizeof(uint16_t));
        __sync_synchronize();
    }

    // -----------------------------------------------------------------------
    // decode_step — one TDT step (begin/wait per step)
    // -----------------------------------------------------------------------

    void decode_step(int enc_frame_idx, int prev_token) {
        gpu->begin_commands();

        // 1. Embed + concat with h[0]
        kl.embed_concat(w->embed_w, prev_token, lstm_h[0].gpu, lstm_input.gpu, CL_D_PRED);

        // 2. LSTM layer 0
        gnt_bias(lstm_input.gpu, 1, 2 * CL_D_PRED, lstm_combined_w[0].gpu,
                 4 * CL_D_PRED, lstm_combined_bias[0].gpu, lstm_gates.gpu);
        kl.lstm_cell(lstm_gates.gpu, lstm_c[0].gpu, lstm_h_out[0].gpu, lstm_c_out[0].gpu, CL_D_PRED);

        // 3. Concat h_out[0] with h[1]
        kl.concat_vectors(lstm_h_out[0].gpu, lstm_h[1].gpu, lstm_input.gpu, CL_D_PRED);

        // 4. LSTM layer 1
        gnt_bias(lstm_input.gpu, 1, 2 * CL_D_PRED, lstm_combined_w[1].gpu,
                 4 * CL_D_PRED, lstm_combined_bias[1].gpu, lstm_gates.gpu);
        kl.lstm_cell(lstm_gates.gpu, lstm_c[1].gpu, lstm_h_out[1].gpu, lstm_c_out[1].gpu, CL_D_PRED);

        // 5. Joint network
        uint64_t enc_proj_t = enc_proj_all.gpu + (size_t)enc_frame_idx * CL_D_JOINT * sizeof(uint16_t);
        gnn_bias(lstm_h_out[1].gpu, 1, CL_D_PRED, w->dec_proj_w, CL_D_JOINT, w->dec_proj_b, dec_proj_buf.gpu);
        kl.add_relu(enc_proj_t, dec_proj_buf.gpu, joint_act.gpu, CL_D_JOINT);
        gnn_bias(joint_act.gpu, 1, CL_D_JOINT, w->out_proj_w, CL_D_OUTPUT, w->out_proj_b, joint_out.gpu);

        // 6. Dual argmax
        kl.dual_argmax(joint_out.gpu, argmax_out.gpu, CL_N_VOCAB, CL_D_OUTPUT);

        gpu->wait_kernel();
    }

    // -----------------------------------------------------------------------
    // decoder_commit — swap LSTM states
    // -----------------------------------------------------------------------

    void decoder_commit() {
        std::swap(lstm_h[0], lstm_h_out[0]);
        std::swap(lstm_c[0], lstm_c_out[0]);
        std::swap(lstm_h[1], lstm_h_out[1]);
        std::swap(lstm_c[1], lstm_c_out[1]);
    }
};
