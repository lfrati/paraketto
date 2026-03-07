// conformer.cpp — Weight loading + FP8 inference for CUDA backend
//
// FP8 E4M3 weight quantization at init time, FP8 GEMMs via cublasLt.

#include "conformer.h"
#include "common.h"
#include "kernels.h"
#include "kernels_fp8.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helper: align up
// ---------------------------------------------------------------------------

static size_t align_up(size_t x, size_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Parse header
// ---------------------------------------------------------------------------

static std::vector<TensorDesc> parse_header(const char* header_text, size_t header_len) {
    std::vector<TensorDesc> tensors;
    std::string text(header_text, header_len);
    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        TensorDesc td;
        ls >> td.name >> td.offset >> td.size_bytes >> td.dtype;
        int d;
        while (ls >> d) td.shape.push_back(d);
        tensors.push_back(std::move(td));
    }
    return tensors;
}

// ---------------------------------------------------------------------------
// Weights: pointer assignment (shared by load and upload)
// ---------------------------------------------------------------------------

static void assign_weight_pointers(Weights& w) {
    uint8_t* gpu_base = (uint8_t*)w.gpu_data;

    auto lookup = [&](const std::string& name) -> half* {
        auto it = w.name_to_idx.find(name);
        if (it == w.name_to_idx.end()) return nullptr;
        return (half*)(gpu_base + w.tensors[it->second].offset);
    };

    // Subsampling (pre_encode)
    for (int i : {0, 2, 3, 5, 6}) {
        std::string pre = "encoder/pre_encode.conv." + std::to_string(i);
        w.sub_conv[i].weight = lookup(pre + ".weight");
        w.sub_conv[i].bias   = lookup(pre + ".bias");
    }
    w.sub_out_w = lookup("encoder/pre_encode.out.weight");
    w.sub_out_b = lookup("encoder/pre_encode.out.bias");

    // Conformer blocks (x24)
    for (int i = 0; i < 24; i++) {
        auto& blk = w.blocks[i];
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

    // Decoder: embedding + LSTM
    w.embed_w = lookup("decoder/decoder.prediction.embed.weight");
    w.lstm0_w_ih = lookup("decoder/decoder.dec_rnn.lstm.weight_ih");
    w.lstm0_w_hh = lookup("decoder/decoder.dec_rnn.lstm.weight_hh");
    w.lstm0_bias = lookup("decoder/decoder.dec_rnn.lstm.bias");
    w.lstm1_w_ih = lookup("decoder/decoder.dec_rnn.lstm.1.weight_ih");
    w.lstm1_w_hh = lookup("decoder/decoder.dec_rnn.lstm.1.weight_hh");
    w.lstm1_bias = lookup("decoder/decoder.dec_rnn.lstm.1.bias");

    // Joint network
    w.enc_proj_w = lookup("decoder/joint.enc.weight");
    w.enc_proj_b = lookup("decoder/joint.enc.bias");
    w.dec_proj_w = lookup("decoder/joint.pred.weight");
    w.dec_proj_b = lookup("decoder/joint.pred.bias");
    w.out_proj_w = lookup("decoder/joint.joint_net.joint_net.2.weight");
    w.out_proj_b = lookup("decoder/joint.joint_net.2.bias");
}

// ---------------------------------------------------------------------------
// Weights::prefetch — CPU only, no CUDA. mmap + populate pages + parse header.
// ---------------------------------------------------------------------------

Weights Weights::prefetch(const std::string& path) {
    Weights w;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open weights: %s\n", path.c_str());
        std::exit(1);
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", path.c_str());
        std::exit(1);
    }
    madvise(mapped, file_size, MADV_SEQUENTIAL);
    const uint8_t* base = (const uint8_t*)mapped;

    // Parse file header
    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != PRKT_MAGIC) {
        fprintf(stderr, "Bad magic in %s: expected PRKT\n", path.c_str());
        munmap(mapped, file_size);
        std::exit(1);
    }

    uint32_t version;
    memcpy(&version, base + 4, 4);
    if (version != PRKT_VERSION) {
        fprintf(stderr, "Unsupported weight file version %u (expected %u)\n", version, PRKT_VERSION);
        munmap(mapped, file_size);
        std::exit(1);
    }

    uint64_t header_len;
    memcpy(&header_len, base + 8, 8);
    w.tensors = parse_header((const char*)(base + 16), header_len);

    for (size_t i = 0; i < w.tensors.size(); i++)
        w.name_to_idx[w.tensors[i].name] = i;

    size_t header_end = 16 + header_len;
    size_t data_start = align_up(header_end, HEADER_ALIGN);

    size_t total_data = 0;
    for (auto& td : w.tensors) {
        size_t end = td.offset + td.size_bytes;
        if (end > total_data) total_data = end;
    }
    if (!w.tensors.empty()) {
        auto& last = w.tensors.back();
        total_data = std::max(total_data, align_up(last.offset + last.size_bytes, 256));
    }

    w.gpu_data_size = total_data;
    w.mmap_ptr = mapped;
    w.mmap_size = file_size;
    w.data_offset = data_start;

    return w;
}

// ---------------------------------------------------------------------------
// Weights::upload — cudaMalloc + cudaMemcpy from prefetched mmap, assign ptrs.
// ---------------------------------------------------------------------------

void Weights::upload(cudaStream_t stream) {
    const uint8_t* base = (const uint8_t*)mmap_ptr;

    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_data_size));
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_data, base + data_offset, gpu_data_size,
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(gpu_data, base + data_offset, gpu_data_size,
                               cudaMemcpyHostToDevice));
    }

    munmap(mmap_ptr, mmap_size);
    mmap_ptr = nullptr;
    mmap_size = 0;
    data_offset = 0;

    assign_weight_pointers(*this);
}

// ---------------------------------------------------------------------------
// Weights::load — convenience: prefetch + upload in one call.
// ---------------------------------------------------------------------------

Weights Weights::load(const std::string& path, cudaStream_t stream) {
    Weights w = prefetch(path);
    w.upload(stream);
    return w;
}

// ---------------------------------------------------------------------------
// Weights::free
// ---------------------------------------------------------------------------

void Weights::free() {
    if (gpu_data) {
        cudaFree(gpu_data);
        gpu_data = nullptr;
        gpu_data_size = 0;
    }
}

// ---------------------------------------------------------------------------
// Weights::get
// ---------------------------------------------------------------------------

half* Weights::get(const std::string& name) const {
    auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) return nullptr;
    return (half*)((uint8_t*)gpu_data + tensors[it->second].offset);
}

// ---------------------------------------------------------------------------
// Weights::print_info
// ---------------------------------------------------------------------------

void Weights::print_info() const {
    fprintf(stderr, "weights: %zu tensors, %.1f MB GPU\n",
            tensors.size(), gpu_data_size / (1024.0 * 1024.0));

    // Check key struct fields are populated
    int missing = 0;
    auto check = [&](const char* label, const half* ptr) {
        if (!ptr) {
            fprintf(stderr, "  WARNING: %s not found in weight file\n", label);
            missing++;
        }
    };

    // Subsampling
    check("sub_conv[0].weight", sub_conv[0].weight);
    check("sub_conv[6].weight", sub_conv[6].weight);
    check("sub_out_w", sub_out_w);

    // Conformer blocks (spot check first and last)
    check("blocks[0].ff1_w1", blocks[0].ff1_w1);
    check("blocks[0].q_w", blocks[0].q_w);
    check("blocks[0].k_w", blocks[0].k_w);
    check("blocks[0].v_w", blocks[0].v_w);
    check("blocks[0].conv_dw_w", blocks[0].conv_dw_w);
    check("blocks[23].ff2_w2", blocks[23].ff2_w2);
    check("blocks[23].final_ln_w", blocks[23].final_ln_w);

    // Decoder
    check("embed_w", embed_w);
    check("lstm0_w_ih", lstm0_w_ih);
    check("lstm0_bias", lstm0_bias);
    check("lstm1_w_ih", lstm1_w_ih);
    check("lstm1_bias", lstm1_bias);

    // Joint
    check("enc_proj_w", enc_proj_w);
    check("dec_proj_w", dec_proj_w);
    check("out_proj_w", out_proj_w);
    check("out_proj_b", out_proj_b);

    if (missing > 0) {
        fprintf(stderr, "  %d key weights missing — run 'make inspect-onnx' to check names\n", missing);
    } else {
        fprintf(stderr, "  all key weights mapped successfully\n");
    }
}

// =========================================================================
// CudaModel — encoder + decoder forward pass
// =========================================================================

// cuBLAS error checking
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t stat = (call);                                         \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                    \
                    __FILE__, __LINE__, (int)stat);                            \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// CudaModel::init
// ---------------------------------------------------------------------------

void CudaModel::init(const Weights& weights, cudaStream_t s, int max_mel_frames) {
    w = &weights;
    stream = s;
    T_max = max_mel_frames / 8 + 10;  // encoder frames after 8x downsampling

    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasLtCreate(&cublaslt));

    // Shared workspace for cublasLt algorithm search and matmul
    lt_workspace_size = 32 * 1024 * 1024;  // 32MB workspace
    CUDA_CHECK(cudaMalloc(&lt_workspace, lt_workspace_size));
    CUBLAS_CHECK(cublasSetWorkspace(cublas, lt_workspace, lt_workspace_size));

    // --- Single pooled GPU allocation for all inference buffers ---
    // Replaces ~35 individual cudaMalloc calls with one.
    int max_sub = SUB_CHANNELS * (max_mel_frames / 2 + 1) * 65;
    size_t mel_fp32_elems = 128 * max_mel_frames;  // stored as float, not half

    // Compute sizes (in half elements) and assign pointers from pool
    size_t sizes[] = {
        (size_t)max_sub,                        // sub_buf[0]
        (size_t)max_sub,                        // sub_buf[1]
        (size_t)(128 * max_mel_frames),         // mel_fp16
        (size_t)(T_max * D_MODEL),              // x
        (size_t)(T_max * D_MODEL),              // ln_out
        (size_t)(T_max * D_FF),                 // ff_mid
        (size_t)(T_max * D_MODEL),              // ff_out
        (size_t)(T_max * 3 * D_MODEL),          // qkv
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // q
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // k
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // v
        (size_t)((2 * T_max) * D_MODEL),        // pos_enc
        (size_t)((2 * T_max) * D_MODEL),        // pos_proj
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // q_u
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // q_v_buf
        (size_t)(N_HEADS * T_max * T_max),      // scores
        (size_t)(N_HEADS * T_max * (2*T_max)),  // pos_scores
        (size_t)(N_HEADS * T_max * HEAD_DIM),   // attn_out
        (size_t)(T_max * D_MODEL),              // mhsa_out
        (size_t)(T_max * D_CONV_PW),            // conv_mid
        (size_t)(T_max * D_MODEL),              // conv_glu
        (size_t)(T_max * D_MODEL),              // conv_dw
        (size_t)(2 * D_PRED),                   // lstm_input (concat buffer)
        (size_t)(4 * D_PRED),                   // lstm_gates
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_h[0], lstm_h[1]
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_c[0], lstm_c[1]
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_h_out[0], lstm_h_out[1]
        (size_t)D_PRED, (size_t)D_PRED,         // lstm_c_out[0], lstm_c_out[1]
        (size_t)(T_max * D_JOINT),              // enc_proj_all (precomputed)
        (size_t)D_JOINT, (size_t)D_JOINT,       // dec_proj_buf, joint_act
        (size_t)D_OUTPUT,                       // joint_out
        // Pre-combined LSTM weights
        (size_t)(4 * D_PRED * 2 * D_PRED),     // lstm_combined_w[0]
        (size_t)(4 * D_PRED * 2 * D_PRED),     // lstm_combined_w[1]
        (size_t)(4 * D_PRED),                   // lstm_combined_bias[0]
        (size_t)(4 * D_PRED),                   // lstm_combined_bias[1]
    };
    constexpr int N_BUFS = sizeof(sizes) / sizeof(sizes[0]);

    // mel_fp32 is float not half — allocate separately; everything else is half.
    size_t total_half = 0;
    for (int i = 0; i < N_BUFS; i++) total_half += sizes[i];

    // Also need: QKV weights (24 blocks), mel_fp32
    size_t qkv_total = (size_t)N_BLOCKS * D_MODEL * 3 * D_MODEL;

    char* pool;
    // Add generous alignment padding (256 bytes per buffer)
    int total_bufs = N_BUFS + N_BLOCKS + 2;  // +mel_fp32 +argmax_out
    size_t pool_bytes = (total_half + qkv_total) * sizeof(half)
                      + mel_fp32_elems * sizeof(float)
                      + 2 * sizeof(int)  // argmax_out
                      + (size_t)total_bufs * 256;  // alignment padding
    CUDA_CHECK(cudaMalloc(&pool, pool_bytes));
    gpu_pool = pool;  // save for free()

    // Assign half* pointers from pool (256-byte aligned for tensor core ops)
    constexpr size_t ALIGN = 256;
    constexpr size_t HALF_ALIGN = ALIGN / sizeof(half);  // 128 halfs
    char* pool_base = pool;
    auto take = [&](size_t n) -> half* {
        // Align pool pointer up to ALIGN boundary
        pool = (char*)(((uintptr_t)pool + ALIGN - 1) & ~(ALIGN - 1));
        half* r = (half*)pool;
        pool += n * sizeof(half);
        return r;
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
    dec_proj_buf = take(sizes[33]);  joint_act = take(sizes[34]);
    joint_out    = take(sizes[35]);
    lstm_combined_w[0]    = take(sizes[36]);
    lstm_combined_w[1]    = take(sizes[37]);
    lstm_combined_bias[0] = take(sizes[38]);
    lstm_combined_bias[1] = take(sizes[39]);

    // QKV weight blocks from pool
    for (int b = 0; b < N_BLOCKS; b++) {
        qkv_w[b] = take(D_MODEL * 3 * D_MODEL);
    }

    // mel_fp32 (float*) from pool, aligned
    pool = (char*)(((uintptr_t)pool + ALIGN - 1) & ~(ALIGN - 1));
    mel_fp32 = (float*)pool;
    pool += mel_fp32_elems * sizeof(float);

    // argmax_out (int[2]) from pool, aligned
    pool = (char*)(((uintptr_t)pool + ALIGN - 1) & ~(ALIGN - 1));
    argmax_out = (int*)pool;
    pool += 2 * sizeof(int);

    // Pre-concatenate QKV weights using cudaMemcpy2D (72 calls vs 73K row-by-row)
    for (int b = 0; b < N_BLOCKS; b++) {
        size_t dst_pitch = 3 * D_MODEL * sizeof(half);
        size_t src_pitch = D_MODEL * sizeof(half);
        size_t width = D_MODEL * sizeof(half);
        CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_w[b], dst_pitch,
            weights.blocks[b].q_w, src_pitch,
            width, D_MODEL, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_w[b] + D_MODEL, dst_pitch,
            weights.blocks[b].k_w, src_pitch,
            width, D_MODEL, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(
            qkv_w[b] + 2 * D_MODEL, dst_pitch,
            weights.blocks[b].v_w, src_pitch,
            width, D_MODEL, cudaMemcpyDeviceToDevice, stream));
    }

    // Pre-combine LSTM weights: W_combined[4*D, 2*D] = [W_ih | W_hh] per layer
    // Same cudaMemcpy2D pattern as QKV concatenation above.
    {
        const half* w_ih[2] = { weights.lstm0_w_ih, weights.lstm1_w_ih };
        const half* w_hh[2] = { weights.lstm0_w_hh, weights.lstm1_w_hh };
        const half* bias[2] = { weights.lstm0_bias, weights.lstm1_bias };
        size_t dst_pitch = 2 * D_PRED * sizeof(half);
        size_t src_pitch = D_PRED * sizeof(half);
        size_t width = D_PRED * sizeof(half);
        for (int layer = 0; layer < 2; layer++) {
            // Left half: W_ih
            CUDA_CHECK(cudaMemcpy2DAsync(
                lstm_combined_w[layer], dst_pitch,
                w_ih[layer], src_pitch,
                width, 4 * D_PRED, cudaMemcpyDeviceToDevice, stream));
            // Right half: W_hh
            CUDA_CHECK(cudaMemcpy2DAsync(
                lstm_combined_w[layer] + D_PRED, dst_pitch,
                w_hh[layer], src_pitch,
                width, 4 * D_PRED, cudaMemcpyDeviceToDevice, stream));
            // Pre-add biases: combined_bias = b_ih + b_hh
            residual_add_fp16(bias[layer], bias[layer] + 4 * D_PRED,
                              lstm_combined_bias[layer], 4 * D_PRED, 1.0f, stream);
        }
    }

    // Pre-compute position encoding for T_max (reused via offset for any T <= T_max)
    generate_pos_encoding_gpu(pos_enc, T_max, D_MODEL, stream);

    // -----------------------------------------------------------------------
    // FP8 weight quantization — quantize all lt_gemm weight matrices to E4M3
    // -----------------------------------------------------------------------
    {
        // Compute total FP8 pool size (1 byte per element)
        size_t per_block = (size_t)D_MODEL * 3 * D_MODEL    // qkv_w
                         + (size_t)D_MODEL * D_FF            // ff1_w1
                         + (size_t)D_FF * D_MODEL            // ff1_w2
                         + (size_t)D_MODEL * D_FF            // ff2_w1
                         + (size_t)D_FF * D_MODEL            // ff2_w2
                         + (size_t)D_MODEL * D_MODEL          // pos_w
                         + (size_t)D_MODEL * D_MODEL          // out_w
                         + (size_t)D_CONV_PW * D_MODEL        // conv_pw1_w
                         + (size_t)D_MODEL * D_MODEL;         // conv_pw2_w
        size_t fp8_total = per_block * N_BLOCKS
                         + (size_t)SUB_CHANNELS * 16 * D_MODEL  // sub_out_w [4096,1024]
                         + (size_t)D_MODEL * D_JOINT              // enc_proj_w
                         + (size_t)4 * D_PRED * 2 * D_PRED * 2   // lstm_combined_w[0,1]
                         + (size_t)D_PRED * D_JOINT               // dec_proj_w
                         + (size_t)D_JOINT * D_OUTPUT;            // out_proj_w
        // Add alignment padding (256 per weight) + scales + act buffer
        int n_fp8_ptrs = N_BLOCKS * 9 + 6;
        size_t fp8_pool_bytes = fp8_total + (size_t)n_fp8_ptrs * 256
                              + N_FP8_SCALES * sizeof(float) + 256
                              + (size_t)T_max * D_FF + 256   // act_buf
                              + sizeof(float) + 256;          // act_scale

        char* fp8p;
        CUDA_CHECK(cudaMalloc(&fp8p, fp8_pool_bytes));
        fp8_pool = fp8p;

        auto take8 = [&](size_t n) -> uint8_t* {
            fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
            uint8_t* r = (uint8_t*)fp8p;
            fp8p += n;
            return r;
        };

        // Per-block FP8 weights
        for (int b = 0; b < N_BLOCKS; b++) {
            fp8_qkv_w[b]     = take8(D_MODEL * 3 * D_MODEL);
            fp8_ff1_w1[b]    = take8(D_MODEL * D_FF);
            fp8_ff1_w2[b]    = take8(D_FF * D_MODEL);
            fp8_ff2_w1[b]    = take8(D_MODEL * D_FF);
            fp8_ff2_w2[b]    = take8(D_FF * D_MODEL);
            fp8_pos_w[b]     = take8(D_MODEL * D_MODEL);
            fp8_out_w[b]     = take8(D_MODEL * D_MODEL);
            fp8_conv_pw1_w[b] = take8(D_CONV_PW * D_MODEL);
            fp8_conv_pw2_w[b] = take8(D_MODEL * D_MODEL);
        }
        fp8_sub_out_w        = take8(SUB_CHANNELS * 16 * D_MODEL);
        fp8_enc_proj_w       = take8(D_MODEL * D_JOINT);
        fp8_lstm_combined_w[0] = take8(4 * D_PRED * 2 * D_PRED);
        fp8_lstm_combined_w[1] = take8(4 * D_PRED * 2 * D_PRED);
        fp8_dec_proj_w       = take8(D_PRED * D_JOINT);
        fp8_out_proj_w       = take8(D_JOINT * D_OUTPUT);

        // Scales array
        fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
        fp8_scales = (float*)fp8p;
        fp8p += N_FP8_SCALES * sizeof(float);

        // Activation scratch
        fp8_act_buf = take8(T_max * D_FF);
        fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
        fp8_amax_buf = (int*)fp8p;
        fp8p += sizeof(int);

        // Per-site activation scale cache
        fp8p = (char*)(((uintptr_t)fp8p + 255) & ~(uintptr_t)255);
        fp8_act_site_scales = (float*)fp8p;
        fp8p += N_FP8_ACT_SITES * sizeof(float);


        // Quantize all weight matrices
        int si = 0;  // scale index
        for (int b = 0; b < N_BLOCKS; b++) {
            auto& blk = weights.blocks[b];
            quantize_absmax_fp16_to_fp8(qkv_w[b],        fp8_qkv_w[b],      &fp8_scales[si++], D_MODEL * 3 * D_MODEL, fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.ff1_w1,      fp8_ff1_w1[b],     &fp8_scales[si++], D_MODEL * D_FF,        fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.ff1_w2,      fp8_ff1_w2[b],     &fp8_scales[si++], D_FF * D_MODEL,        fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.ff2_w1,      fp8_ff2_w1[b],     &fp8_scales[si++], D_MODEL * D_FF,        fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.ff2_w2,      fp8_ff2_w2[b],     &fp8_scales[si++], D_FF * D_MODEL,        fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.pos_w,       fp8_pos_w[b],      &fp8_scales[si++], D_MODEL * D_MODEL,     fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.out_w,       fp8_out_w[b],      &fp8_scales[si++], D_MODEL * D_MODEL,     fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.conv_pw1_w,  fp8_conv_pw1_w[b], &fp8_scales[si++], D_CONV_PW * D_MODEL,   fp8_amax_buf, stream);
            quantize_absmax_fp16_to_fp8(blk.conv_pw2_w,  fp8_conv_pw2_w[b], &fp8_scales[si++], D_MODEL * D_MODEL,     fp8_amax_buf, stream);
        }
        quantize_absmax_fp16_to_fp8(weights.sub_out_w,   fp8_sub_out_w,     &fp8_scales[si++], SUB_CHANNELS * 16 * D_MODEL, fp8_amax_buf, stream);
        quantize_absmax_fp16_to_fp8(weights.enc_proj_w,  fp8_enc_proj_w,    &fp8_scales[si++], D_MODEL * D_JOINT,           fp8_amax_buf, stream);
        quantize_absmax_fp16_to_fp8(lstm_combined_w[0],  fp8_lstm_combined_w[0], &fp8_scales[si++], 4 * D_PRED * 2 * D_PRED, fp8_amax_buf, stream);
        quantize_absmax_fp16_to_fp8(lstm_combined_w[1],  fp8_lstm_combined_w[1], &fp8_scales[si++], 4 * D_PRED * 2 * D_PRED, fp8_amax_buf, stream);
        quantize_absmax_fp16_to_fp8(weights.dec_proj_w,  fp8_dec_proj_w,    &fp8_scales[si++], D_PRED * D_JOINT,            fp8_amax_buf, stream);
        quantize_absmax_fp16_to_fp8(weights.out_proj_w,  fp8_out_proj_w,    &fp8_scales[si++], D_JOINT * D_OUTPUT,          fp8_amax_buf, stream);
        assert(si == N_FP8_SCALES);

        fprintf(stderr, "  FP8 quantized %d weight matrices (%.0f MB)\n",
                N_FP8_SCALES, fp8_total / (1024.0 * 1024.0));
    }

    // Allocate CUTLASS workspace (queried for max GEMM size)
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// cublasLt GEMM algorithm cache (forward declaration for free())
// ---------------------------------------------------------------------------

struct GemmKey {
    int m, n, k;
    cublasOperation_t opA, opB;
    bool accumulate;
    bool has_bias;
    bool operator==(const GemmKey& o) const {
        return m == o.m && n == o.n && k == o.k && opA == o.opA && opB == o.opB
            && accumulate == o.accumulate && has_bias == o.has_bias;
    }
};
struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const {
        size_t h = std::hash<int>()(k.m);
        h ^= std::hash<int>()(k.n) * 2654435761ULL;
        h ^= std::hash<int>()(k.k) * 40343;
        h ^= std::hash<int>()(k.opA) * 73;
        h ^= std::hash<int>()(k.opB) * 131;
        h ^= std::hash<int>()(k.accumulate) * 257;
        h ^= std::hash<int>()(k.has_bias) * 521;
        return h;
    }
};
struct GemmPlan {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
    cublasLtMatmulAlgo_t algo;
    bool valid;
};
static std::unordered_map<GemmKey, GemmPlan, GemmKeyHash> gemm_cache;

// FP8 GEMM plan cache (forward declaration for free())
struct Fp8GemmPlan {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
    cublasLtMatmulAlgo_t algo;
    bool valid;
};
static std::unordered_map<GemmKey, Fp8GemmPlan, GemmKeyHash> fp8_gemm_cache;

// ---------------------------------------------------------------------------
// CudaModel::free
// ---------------------------------------------------------------------------

void CudaModel::free() {
    if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
    if (cublaslt) { cublasLtDestroy(cublaslt); cublaslt = nullptr; }
    // Clean up cached cublasLt plans
    for (auto& [key, plan] : gemm_cache) {
        cublasLtMatmulDescDestroy(plan.matmul_desc);
        cublasLtMatrixLayoutDestroy(plan.layout_A);
        cublasLtMatrixLayoutDestroy(plan.layout_B);
        cublasLtMatrixLayoutDestroy(plan.layout_C);
    }
    gemm_cache.clear();
    for (auto& [key, plan] : fp8_gemm_cache) {
        cublasLtMatmulDescDestroy(plan.matmul_desc);
        cublasLtMatrixLayoutDestroy(plan.layout_A);
        cublasLtMatrixLayoutDestroy(plan.layout_B);
        cublasLtMatrixLayoutDestroy(plan.layout_C);
    }
    fp8_gemm_cache.clear();
    if (lt_workspace) { cudaFree(lt_workspace); lt_workspace = nullptr; }

    // All inference buffers are carved from a single pooled allocation
    if (gpu_pool) { cudaFree(gpu_pool); gpu_pool = nullptr; }
    if (fp8_pool) { cudaFree(fp8_pool); fp8_pool = nullptr; }
}

// ---------------------------------------------------------------------------
// cublasLt GEMM with algorithm caching
// ---------------------------------------------------------------------------

static GemmPlan& get_gemm_plan(cublasLtHandle_t lt, void* workspace, size_t ws_size,
                                int col_m, int col_n, int col_k,
                                int ldA, int ldB, int ldC,
                                cublasOperation_t opA, cublasOperation_t opB,
                                bool accumulate, bool has_bias = false) {
    GemmKey key{col_m, col_n, col_k, opA, opB, accumulate, has_bias};
    auto it = gemm_cache.find(key);
    if (it != gemm_cache.end()) return it->second;

    GemmPlan plan;
    plan.valid = false;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.matmul_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    if (has_bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    }

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_A, CUDA_R_16F,
        opA == CUBLAS_OP_N ? col_m : col_k,
        opA == CUBLAS_OP_N ? col_k : col_m, ldA));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_B, CUDA_R_16F,
        opB == CUBLAS_OP_N ? col_k : col_n,
        opB == CUBLAS_OP_N ? col_n : col_k, ldB));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_C, CUDA_R_16F, col_m, col_n, ldC));

    // Search for best algorithm
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

    cublasLtMatmulHeuristicResult_t results[8];
    int n_results = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, plan.matmul_desc,
        plan.layout_A, plan.layout_B, plan.layout_C, plan.layout_C,
        pref, 8, results, &n_results));
    cublasLtMatmulPreferenceDestroy(pref);

    if (n_results > 0) {
        plan.algo = results[0].algo;
        plan.valid = true;
    }

    return gemm_cache.emplace(key, plan).first->second;
}

// Core cublasLt GEMM call using cached plan
static void lt_gemm(cublasLtHandle_t lt, cublasHandle_t h,
                      void* workspace, size_t ws_size, cudaStream_t stream,
                      cublasOperation_t opA, cublasOperation_t opB,
                      int col_m, int col_n, int col_k,
                      const half* A, int ldA,
                      const half* B, int ldB,
                      half* C, int ldC,
                      bool accumulate, const half* bias = nullptr) {
    half alpha = __float2half(1.0f);
    half beta = __float2half(accumulate ? 1.0f : 0.0f);

    GemmPlan& plan = get_gemm_plan(lt, workspace, ws_size,
                                     col_m, col_n, col_k, ldA, ldB, ldC,
                                     opA, opB, accumulate, bias != nullptr);
    if (plan.valid) {
        if (bias) {
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        }
        CUBLAS_CHECK(cublasLtMatmul(lt, plan.matmul_desc,
            &alpha, A, plan.layout_A, B, plan.layout_B,
            &beta, C, plan.layout_C, C, plan.layout_C,
            &plan.algo, workspace, ws_size, stream));
    } else {
        // Fallback to cublasGemmEx (no bias epilogue support)
        CUBLAS_CHECK(cublasGemmEx(h, opA, opB,
            col_m, col_n, col_k, &alpha, A, CUDA_R_16F, ldA,
            B, CUDA_R_16F, ldB, &beta, C, CUDA_R_16F, ldC,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

// Y[m,n] = X[m,k] @ W[k,n]   (ONNX MatMul convention: W is [input, output])
static void gemm_nn(cublasLtHandle_t lt, cublasHandle_t h,
                     void* ws, size_t ws_size, cudaStream_t stream,
                     const half* X, int m, int k,
                     const half* W, int n, half* Y) {
    // Row-major: Y[m,n] = X[m,k] @ W[k,n]
    // Col-major: Y'[n,m] = W'[n,k] @ X'[k,m]
    lt_gemm(lt, h, ws, ws_size, stream, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k, W, n, X, k, Y, n, false);
}

// Y[m,n] = X[m,k] @ W[k,n] + bias[n]  (ONNX MatMul + bias epilogue)
static void gemm_nn_bias(cublasLtHandle_t lt, cublasHandle_t h,
                          void* ws, size_t ws_size, cudaStream_t stream,
                          const half* X, int m, int k,
                          const half* W, int n, const half* bias, half* Y) {
    lt_gemm(lt, h, ws, ws_size, stream, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k, W, n, X, k, Y, n, false, bias);
}

// Y[m,n] = X[m,k] @ W[n,k]^T  (Conv/PyTorch convention: W is [output, input])
static void gemm_nt(cublasLtHandle_t lt, cublasHandle_t h,
                     void* ws, size_t ws_size, cudaStream_t stream,
                     const half* X, int m, int k,
                     const half* W, int n, half* Y) {
    // Row-major: Y[m,n] = X[m,k] @ W[n,k]^T
    // Col-major: Y'[n,m] = W_T'[n,k] @ X'[k,m], W_T = CUBLAS_OP_T on W[k,n]-col
    lt_gemm(lt, h, ws, ws_size, stream, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, W, k, X, k, Y, n, false);
}

// Y += X @ W[n,k]^T  (accumulate, Conv/PyTorch convention)
static void gemm_nt_accum(cublasLtHandle_t lt, cublasHandle_t h,
                            void* ws, size_t ws_size, cudaStream_t stream,
                            const half* X, int m, int k,
                            const half* W, int n, half* Y) {
    lt_gemm(lt, h, ws, ws_size, stream, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, W, k, X, k, Y, n, true);
}

// Y[m,n] = X[m,k] @ W[n,k]^T + bias[n]  (NT with bias epilogue)
static void gemm_nt_bias(cublasLtHandle_t lt, cublasHandle_t h,
                           void* ws, size_t ws_size, cudaStream_t stream,
                           const half* X, int m, int k,
                           const half* W, int n, const half* bias, half* Y) {
    lt_gemm(lt, h, ws, ws_size, stream, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, W, k, X, k, Y, n, false, bias);
}

// Y += X @ W[n,k]^T + bias[n]  (NT accumulate with bias epilogue)
static void gemm_nt_accum_bias(cublasLtHandle_t lt, cublasHandle_t h,
                                 void* ws, size_t ws_size, cudaStream_t stream,
                                 const half* X, int m, int k,
                                 const half* W, int n, const half* bias, half* Y) {
    lt_gemm(lt, h, ws, ws_size, stream, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, W, k, X, k, Y, n, true, bias);
}

// ---------------------------------------------------------------------------
// FP8 cublasLt GEMM — FP8 inputs, FP16 output, FP32 accumulation
// ---------------------------------------------------------------------------

static Fp8GemmPlan& get_fp8_gemm_plan(cublasLtHandle_t lt, void* workspace, size_t ws_size,
                                        int col_m, int col_n, int col_k,
                                        int ldA, int ldB, int ldC,
                                        cublasOperation_t opA, cublasOperation_t opB,
                                        bool accumulate) {
    GemmKey key{col_m, col_n, col_k, opA, opB, accumulate, false};
    auto it = fp8_gemm_cache.find(key);
    if (it != fp8_gemm_cache.end()) return it->second;

    Fp8GemmPlan plan;
    plan.valid = false;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // Note: bias epilogue is NOT supported with FP8 on Blackwell — bias added separately

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_A, CUDA_R_8F_E4M3,
        opA == CUBLAS_OP_N ? col_m : col_k,
        opA == CUBLAS_OP_N ? col_k : col_m, ldA));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_B, CUDA_R_8F_E4M3,
        opB == CUBLAS_OP_N ? col_k : col_n,
        opB == CUBLAS_OP_N ? col_n : col_k, ldB));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_C, CUDA_R_16F, col_m, col_n, ldC));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

    cublasLtMatmulHeuristicResult_t results[8];
    int n_results = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, plan.matmul_desc,
        plan.layout_A, plan.layout_B, plan.layout_C, plan.layout_C,
        pref, 8, results, &n_results);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st == CUBLAS_STATUS_SUCCESS && n_results > 0) {
        plan.algo = results[0].algo;
        plan.valid = true;
    } else {
        fprintf(stderr, "FP8 heuristic: m=%d n=%d k=%d opA=%d opB=%d => status=%d n_results=%d\n",
                col_m, col_n, col_k, (int)opA, (int)opB, (int)st, n_results);
    }

    return fp8_gemm_cache.emplace(key, plan).first->second;
}

// FP8 GEMM: quantizes activation B, uses pre-quantized weight A.
// site_scale: per-activation-site cached scale (on GPU). On calibration pass,
//   absmax is computed and written here. On subsequent passes, reused as-is.
static void lt_gemm_fp8(cublasLtHandle_t lt, cublasHandle_t h,
                          void* workspace, size_t ws_size, cudaStream_t stream,
                          cublasOperation_t opA, cublasOperation_t opB,
                          int col_m, int col_n, int col_k,
                          const uint8_t* A_fp8, int ldA, const float* a_scale,
                          const half* B_fp16, int ldB,
                          half* C, int ldC,
                          uint8_t* act_buf, float* site_scale, int* amax_buf,
                          bool accumulate, bool calibrated) {
    int b_elems = (opB == CUBLAS_OP_N) ? col_k * col_n : col_n * col_k;
    if (!calibrated) {
        // Calibration: full absmax + quantize, stores scale in *site_scale
        quantize_absmax_fp16_to_fp8(B_fp16, act_buf, site_scale, b_elems, amax_buf, stream);
    } else {
        // Runtime: single-pass quantize using cached scale
        quantize_fp8_static(B_fp16, act_buf, site_scale, b_elems, stream);
    }

    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;

    Fp8GemmPlan& plan = get_fp8_gemm_plan(lt, workspace, ws_size,
                                            col_m, col_n, col_k, ldA, ldB, ldC,
                                            opA, opB, accumulate);
    if (plan.valid) {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &site_scale, sizeof(site_scale)));
        CUBLAS_CHECK(cublasLtMatmul(lt, plan.matmul_desc,
            &alpha, A_fp8, plan.layout_A, act_buf, plan.layout_B,
            &beta, C, plan.layout_C, C, plan.layout_C,
            &plan.algo, workspace, ws_size, stream));
    } else {
        fprintf(stderr, "FP8 GEMM: no valid algorithm found for m=%d n=%d k=%d\n",
                col_m, col_n, col_k);
        std::exit(1);
    }
}

// FP8 wrappers — take per-site cached scale + calibration flag
static void gemm_nn_fp8(cublasLtHandle_t lt, cublasHandle_t h,
                          void* ws, size_t ws_size, cudaStream_t stream,
                          const half* X, int m, int k,
                          const uint8_t* W_fp8, int n, const float* w_scale,
                          half* Y, uint8_t* act_buf, float* site_scale,
                          int* amax_buf, bool calibrated) {
    lt_gemm_fp8(lt, h, ws, ws_size, stream, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k, W_fp8, n, w_scale, X, k, Y, n,
                act_buf, site_scale, amax_buf, false, calibrated);
}

static void gemm_nn_bias_fp8(cublasLtHandle_t lt, cublasHandle_t h,
                               void* ws, size_t ws_size, cudaStream_t stream,
                               const half* X, int m, int k,
                               const uint8_t* W_fp8, int n, const float* w_scale,
                               const half* bias, half* Y, uint8_t* act_buf,
                               float* site_scale, int* amax_buf, bool calibrated) {
    lt_gemm_fp8(lt, h, ws, ws_size, stream, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k, W_fp8, n, w_scale, X, k, Y, n,
                act_buf, site_scale, amax_buf, false, calibrated);
    bias_add_row_fp16(Y, bias, m, n, stream);
}

static void gemm_nt_fp8(cublasLtHandle_t lt, cublasHandle_t h,
                          void* ws, size_t ws_size, cudaStream_t stream,
                          const half* X, int m, int k,
                          const uint8_t* W_fp8, int n, const float* w_scale,
                          half* Y, uint8_t* act_buf, float* site_scale,
                          int* amax_buf, bool calibrated) {
    lt_gemm_fp8(lt, h, ws, ws_size, stream, CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k, W_fp8, k, w_scale, X, k, Y, n,
                act_buf, site_scale, amax_buf, false, calibrated);
}

static void gemm_nt_bias_fp8(cublasLtHandle_t lt, cublasHandle_t h,
                               void* ws, size_t ws_size, cudaStream_t stream,
                               const half* X, int m, int k,
                               const uint8_t* W_fp8, int n, const float* w_scale,
                               const half* bias, half* Y, uint8_t* act_buf,
                               float* site_scale, int* amax_buf, bool calibrated) {
    lt_gemm_fp8(lt, h, ws, ws_size, stream, CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k, W_fp8, k, w_scale, X, k, Y, n,
                act_buf, site_scale, amax_buf, false, calibrated);
    bias_add_row_fp16(Y, bias, m, n, stream);
}

// Batched strided GEMM: C[b,m,n] = A[b,m,k] @ B[b,k,n]
// A is [b,m,k], B is [b,k,n], C is [b,m,n]
static void batched_gemm_nn(cublasHandle_t h,
                             const half* A, const half* B, half* C,
                             int batch, int m, int n, int k,
                             long long strideA, long long strideB, long long strideC) {
    half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    // Row-major C[m,n] = A[m,k] @ B[k,n]
    // Col-major: C'[n,m] = B'[n,k] @ A'[k,m]
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, &alpha, B, CUDA_R_16F, n, strideB,
        A, CUDA_R_16F, k, strideA, &beta, C, CUDA_R_16F, n, strideC,
        batch, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Batched strided GEMM: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T  (B transposed)
// A is [b,m,k], B is [b,n,k], C is [b,m,n]
static void batched_gemm_nt(cublasHandle_t h,
                             const half* A, const half* B, half* C,
                             int batch, int m, int n, int k,
                             long long strideA, long long strideB, long long strideC) {
    half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    // Row-major C[m,n] = A[m,k] @ B[n,k]^T
    // Col-major: C'[n,m] = B_T'[n,k] @ A'[k,m], B_T = CUBLAS_OP_T
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha, B, CUDA_R_16F, k, strideB,
        A, CUDA_R_16F, k, strideA, &beta, C, CUDA_R_16F, n, strideC,
        batch, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Position encoding generation moved to GPU kernel (generate_pos_encoding_gpu)

// ---------------------------------------------------------------------------
// CudaModel::encode
// ---------------------------------------------------------------------------

int CudaModel::encode(const float* mel_fp32_host, int T_mel) {
    // Upload mel from host then delegate to encode_gpu
    CUDA_CHECK(cudaMemcpyAsync(mel_fp32, mel_fp32_host, 128 * T_mel * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    return encode_gpu(T_mel);
}

int CudaModel::encode_gpu(int T_mel) {
    // Local FP16 GEMM helpers (kept for small sub-conv GEMMs and batched GEMMs)
    auto gnn = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nn(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, Y);
    };

    // Local FP8 GEMM helpers — cublasLt FP8 with per-site cached activation scales
    bool cal = fp8_calibrated;

    auto gnn8 = [&](const half* X, int m, int k, const uint8_t* W, int n,
                     const float* ws, int si, half* Y) {
        gemm_nn_fp8(cublaslt, cublas, lt_workspace, lt_workspace_size, stream,
                    X, m, k, W, n, ws, Y, fp8_act_buf,
                    &fp8_act_site_scales[si], fp8_amax_buf, cal);
    };
    auto gnn8_bias = [&](const half* X, int m, int k, const uint8_t* W, int n,
                          const float* ws, int si, const half* bias, half* Y) {
        gemm_nn_bias_fp8(cublaslt, cublas, lt_workspace, lt_workspace_size, stream,
                         X, m, k, W, n, ws, bias, Y, fp8_act_buf,
                         &fp8_act_site_scales[si], fp8_amax_buf, cal);
    };
    auto gnt8 = [&](const half* X, int m, int k, const uint8_t* W, int n,
                     const float* ws, int si, half* Y) {
        gemm_nt_fp8(cublaslt, cublas, lt_workspace, lt_workspace_size, stream,
                    X, m, k, W, n, ws, Y, fp8_act_buf,
                    &fp8_act_site_scales[si], fp8_amax_buf, cal);
    };
    // Weight scale helper: fp8_scales[blk * 9 + offset]
    auto bscale = [&](int blk, int off) -> const float* { return &fp8_scales[blk * 9 + off]; };
    // Activation site index: blk * 9 + offset (0=ff1_w1, 1=ff1_w2, 2=qkv, 3=pos, 4=out,
    //                                          5=conv_pw1, 6=conv_pw2, 7=ff2_w1, 8=ff2_w2)
    auto asite = [](int blk, int off) -> int { return blk * 9 + off; };

    // 1. Cast mel to FP16, then transpose
    cast_fp32_to_fp16(mel_fp32, mel_fp16, 128 * T_mel, stream);
    transpose_fp16(mel_fp16, sub_buf[0], 128, T_mel, stream);

    // 2. Subsampling: [1, T_mel, 128] → [T', 1024]
    //    Small sub-conv GEMMs stay FP16 (tiny weights, not worth FP8 overhead)
    int H = T_mel, W = 128;
    int H2 = (H + 2 * 1 - 3) / 2 + 1;
    int W2 = (W + 2 * 1 - 3) / 2 + 1;
    im2col_2d_fp16(sub_buf[0], ff_mid, 1, H, W, 3, 3, 2, 1, H2, W2, stream);
    gnn(w->sub_conv[0].weight, SUB_CHANNELS, 9, ff_mid, H2 * W2, sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[0].bias, SUB_CHANNELS, H2 * W2, stream);
    H = H2; W = W2;

    conv2d_fp16(sub_buf[1], w->sub_conv[2].weight, w->sub_conv[2].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;
    gnn(w->sub_conv[3].weight, SUB_CHANNELS, SUB_CHANNELS, sub_buf[0], H * W, sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[3].bias, SUB_CHANNELS, H * W, stream);

    conv2d_fp16(sub_buf[1], w->sub_conv[5].weight, w->sub_conv[5].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;
    gnn(w->sub_conv[6].weight, SUB_CHANNELS, SUB_CHANNELS, sub_buf[0], H * W, sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[6].bias, SUB_CHANNELS, H * W, stream);

    int T = H;
    reshape_chw_to_hcw_fp16(sub_buf[1], sub_buf[0], SUB_CHANNELS, T, W, stream);
    // sub_out projection: FP8 (activation site: N_BLOCKS*9 + 0)
    const float* sub_out_scale = &fp8_scales[N_BLOCKS * 9 + 0];
    int sub_out_si = N_BLOCKS * 9 + 0;
    if (w->sub_out_b)
        gnn8_bias(sub_buf[0], T, SUB_CHANNELS * W, fp8_sub_out_w, D_MODEL, sub_out_scale, sub_out_si, w->sub_out_b, x);
    else
        gnn8(sub_buf[0], T, SUB_CHANNELS * W, fp8_sub_out_w, D_MODEL, sub_out_scale, sub_out_si, x);

    // 3. Position encoding
    half* pos_enc_T = pos_enc + (T_max - T) * D_MODEL;
    int pos_len = 2 * T - 1;

    // 4. Conformer blocks — all lt_gemm-based GEMMs use FP8
    for (int blk = 0; blk < N_BLOCKS; blk++) {
        auto& b = w->blocks[blk];

        // --- FF1 (half-step residual) ---
        layer_norm_fp16(x, b.ff1_ln_w, b.ff1_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        gnn8(ln_out, T, D_MODEL, fp8_ff1_w1[blk], D_FF, bscale(blk, 1), asite(blk, 0), ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);

        gnn8(ff_mid, T, D_FF, fp8_ff1_w2[blk], D_MODEL, bscale(blk, 2), asite(blk, 1), ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.mhsa_ln_w, b.mhsa_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        {
            // Fused QKV projection: FP8
            gnn8(ln_out, T, D_MODEL, fp8_qkv_w[blk], 3 * D_MODEL, bscale(blk, 0), asite(blk, 2), qkv);

            half* K_h = k;
            half* V_h = v;
            split_transpose_qkv_bias_fp16(qkv, b.pos_bias_u, b.pos_bias_v,
                                           q_u, q_v_buf, K_h, V_h,
                                           T, N_HEADS, HEAD_DIM, stream);

            // Position encoding projection: FP8
            half* pos_temp = pos_proj;
            gnn8(pos_enc_T, pos_len, D_MODEL, fp8_pos_w[blk], D_MODEL, bscale(blk, 5), asite(blk, 3), pos_temp);

            // Batched GEMMs stay FP16 (dynamic activations, not weight matrices)
            {
                half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pos_len, T, HEAD_DIM,
                    &alpha_h,
                    pos_temp, CUDA_R_16F, D_MODEL, (long long)HEAD_DIM,
                    q_v_buf, CUDA_R_16F, HEAD_DIM, (long long)T * HEAD_DIM,
                    &beta_h,
                    pos_scores, CUDA_R_16F, pos_len, (long long)T * pos_len,
                    N_HEADS, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }

            float scale = 1.0f / sqrtf((float)HEAD_DIM);

            batched_gemm_nt(cublas, q_u, K_h, scores,
                            N_HEADS, T, T, HEAD_DIM,
                            (long long)T * HEAD_DIM, (long long)T * HEAD_DIM, (long long)T * T);
            fused_score_softmax_fp16(scores, pos_scores, scores,
                                      N_HEADS, T, scale, stream);
            batched_gemm_nn(cublas, scores, V_h, attn_out,
                            N_HEADS, T, HEAD_DIM, T,
                            (long long)T * T, (long long)T * HEAD_DIM, (long long)T * HEAD_DIM);
            transpose_0213_fp16(attn_out, ff_out, N_HEADS, T, HEAD_DIM, stream);

            // Output projection: FP8
            gnn8(ff_out, T, D_MODEL, fp8_out_w[blk], D_MODEL, bscale(blk, 6), asite(blk, 4), mhsa_out);
        }

        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.conv_ln_w, b.conv_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        // Pointwise conv1 + GLU: FP8
        gnt8(ln_out, T, D_MODEL, fp8_conv_pw1_w[blk], D_CONV_PW, bscale(blk, 7), asite(blk, 5), conv_mid);
        glu_fp16(conv_mid, conv_glu, T, D_MODEL, stream);

        depthwise_conv1d_k9_silu_fp16(conv_glu, b.conv_dw_w, b.conv_dw_b,
                                       conv_dw, T, D_MODEL, stream);

        // Pointwise conv2: FP8
        gnt8(conv_dw, T, D_MODEL, fp8_conv_pw2_w[blk], D_MODEL, bscale(blk, 8), asite(blk, 6), mhsa_out);

        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.ff2_ln_w, b.ff2_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        // FF2: FP8
        gnn8(ln_out, T, D_MODEL, fp8_ff2_w1[blk], D_FF, bscale(blk, 3), asite(blk, 7), ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);

        gnn8(ff_mid, T, D_FF, fp8_ff2_w2[blk], D_MODEL, bscale(blk, 4), asite(blk, 8), ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.final_ln_w, b.final_ln_b, x, T, D_MODEL, 1e-5f, stream);
    }

    // Encoder projection: FP8 (activation site: N_BLOCKS*9 + 1)
    const float* enc_proj_scale = &fp8_scales[N_BLOCKS * 9 + 1];
    int enc_proj_si = N_BLOCKS * 9 + 1;
    gnn8_bias(x, T, D_MODEL, fp8_enc_proj_w, D_JOINT, enc_proj_scale, enc_proj_si, w->enc_proj_b, enc_proj_all);

    fp8_calibrated = true;
    return T;
}

// ---------------------------------------------------------------------------
// CudaModel::decoder_reset
// ---------------------------------------------------------------------------

void CudaModel::decoder_reset() {
    CUDA_CHECK(cudaMemsetAsync(lstm_h[0], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_c[0], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_h[1], 0, D_PRED * sizeof(half), stream));
    CUDA_CHECK(cudaMemsetAsync(lstm_c[1], 0, D_PRED * sizeof(half), stream));
}

// ---------------------------------------------------------------------------
// CudaModel::decode_step
// ---------------------------------------------------------------------------

half* CudaModel::decode_step(int enc_frame_idx, int prev_token) {
    // Decoder GEMMs use FP16 — cublasLt FP8 doesn't support N=1 (single-vector output)
    auto gnn_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nn_bias(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, bias, Y);
    };
    auto gnt_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nt_bias(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, bias, Y);
    };

    // 1. Embed + concat with h[0] for LSTM0 input: lstm_input = [embed; h[0]]
    embed_concat_fp16(w->embed_w, prev_token, lstm_h[0], lstm_input, D_PRED, stream);

    // 2. LSTM layer 0 — FP16 GEMM with combined [W_ih|W_hh] weights
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[0], 4 * D_PRED, lstm_combined_bias[0], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[0], lstm_h_out[0], lstm_c_out[0], D_PRED, stream);

    // 3. Concat h_out[0] with h[1] for LSTM1 input
    concat_vectors_fp16(lstm_h_out[0], lstm_h[1], lstm_input, D_PRED, stream);

    // 4. LSTM layer 1 — FP16 GEMM
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[1], 4 * D_PRED, lstm_combined_bias[1], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[1], lstm_h_out[1], lstm_c_out[1], D_PRED, stream);

    // 5. Joint network — FP16 GEMMs
    half* enc_proj_t = enc_proj_all + enc_frame_idx * D_JOINT;
    gnn_b(lstm_h_out[1], 1, D_PRED, w->dec_proj_w, D_JOINT, w->dec_proj_b, dec_proj_buf);
    add_relu_fp16(enc_proj_t, dec_proj_buf, joint_act, D_JOINT, stream);
    gnn_b(joint_act, 1, D_JOINT, w->out_proj_w, D_OUTPUT, w->out_proj_b, joint_out);

    return joint_out;
}

// ---------------------------------------------------------------------------
// CudaModel::decoder_commit — swap LSTM state after non-blank token
// ---------------------------------------------------------------------------

void CudaModel::decoder_commit() {
    std::swap(lstm_h[0], lstm_h_out[0]);
    std::swap(lstm_c[0], lstm_c_out[0]);
    std::swap(lstm_h[1], lstm_h_out[1]);
    std::swap(lstm_c[1], lstm_c_out[1]);
}
