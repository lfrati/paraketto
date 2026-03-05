// conformer.cpp — Weight loading for CUDA backend (Phase 1)
//
// Loads a flat binary weight file (produced by scripts/export_weights.py)
// into a single contiguous GPU allocation, then assigns struct field pointers
// by matching tensor names from the file header.

#include "conformer.h"
#include "common.h"
#include "kernels.h"
#include "gemm.h"

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

// ---------------------------------------------------------------------------
// CudaModel::init
// ---------------------------------------------------------------------------

void CudaModel::init(const Weights& weights, cudaStream_t s, int max_mel_frames) {
    w = &weights;
    stream = s;
    T_max = max_mel_frames / 8 + 10;  // encoder frames after 8x downsampling

    gemm_init(stream);

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

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// CudaModel::free
// ---------------------------------------------------------------------------

void CudaModel::free() {
    gemm_free();
    // All inference buffers are carved from a single pooled allocation
    if (gpu_pool) { cudaFree(gpu_pool); gpu_pool = nullptr; }
}

// GEMM calls use gemm.h interface directly — backend selected at link time.

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
    // Local GEMM helpers that capture stream
    auto gnn = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nn(stream, X, m, k, W, n, Y);
    };
    auto gnt = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nt(stream, X, m, k, W, n, Y);
    };
    auto gnn_bias = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nn_bias(stream, X, m, k, W, n, bias, Y);
    };

    // 1. Cast mel to FP16, then transpose
    //    mel_fp32 is [128, T_mel] on GPU
    //    ONNX model transposes to [T_mel, 128] before conv2d
    cast_fp32_to_fp16(mel_fp32, mel_fp16, 128 * T_mel, stream);

    // Transpose mel from [128, T_mel] to [T_mel, 128]
    transpose_fp16(mel_fp16, sub_buf[0], 128, T_mel, stream);
    // sub_buf[0] is now [T_mel, 128] which acts as [C_in=1, H=T_mel, W=128] for conv2d

    // 2. Subsampling: [1, T_mel, 128] → [T', 1024]
    //    conv.0 (1→256, 3x3, s=2) → bias+ReLU
    int H = T_mel, W = 128;
    int H2 = (H + 2 * 1 - 3) / 2 + 1;
    int W2 = (W + 2 * 1 - 3) / 2 + 1;  // 64
    conv2d_fp16(sub_buf[0], w->sub_conv[0].weight, nullptr,
                sub_buf[1], 1, H, W, SUB_CHANNELS, 3, 3, 2, 1, 1, stream);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[0].bias, SUB_CHANNELS, H2 * W2, stream);
    H = H2; W = W2;

    //    conv.2 (depthwise 256, 3x3, s=2) → conv.3 (pointwise 256→256, 1x1) → bias+ReLU
    conv2d_fp16(sub_buf[1], w->sub_conv[2].weight, w->sub_conv[2].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;   // 32
    // Pointwise 1x1 conv = GEMM: out[256, H*W] = weight[256, 256] @ in[256, H*W]
    gnn(/*X=*/w->sub_conv[3].weight, SUB_CHANNELS, SUB_CHANNELS,
        /*W=*/sub_buf[0], H * W, /*Y=*/sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[3].bias, SUB_CHANNELS, H * W, stream);

    //    conv.5 (depthwise 256, 3x3, s=2) → conv.6 (pointwise 256→256, 1x1) → bias+ReLU
    conv2d_fp16(sub_buf[1], w->sub_conv[5].weight, w->sub_conv[5].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;   // T' (encoder frames)
    W = (W + 2 * 1 - 3) / 2 + 1;   // 16
    // Pointwise 1x1 conv = GEMM: out[256, H*W] = weight[256, 256] @ in[256, H*W]
    gnn(/*X=*/w->sub_conv[6].weight, SUB_CHANNELS, SUB_CHANNELS,
        /*W=*/sub_buf[0], H * W, /*Y=*/sub_buf[1]);
    bias_relu_nchw_fp16(sub_buf[1], w->sub_conv[6].bias, SUB_CHANNELS, H * W, stream);

    // Reshape [256, H, 16] → [H, 256*16] = [T', 4096] then linear → [T', 1024]
    // This is [C, H, W] → [H, C*W] (permute(1,0,2) + flatten)
    int T = H;  // encoder frames = time dim after 3x stride-2
    reshape_chw_to_hcw_fp16(sub_buf[1], sub_buf[0], SUB_CHANNELS, T, W, stream);
    // sub_buf[0] is now [T, 4096]
    if (w->sub_out_b)
        gnn_bias(sub_buf[0], T, SUB_CHANNELS * W, w->sub_out_w, D_MODEL, w->sub_out_b, x);
    else
        gnn(sub_buf[0], T, SUB_CHANNELS * W, w->sub_out_w, D_MODEL, x);

    // 3. Position encoding: slice from pre-computed T_max encoding
    //    pos_enc for T is at offset (T_max - T) * D_MODEL in the T_max encoding
    half* pos_enc_T = pos_enc + (T_max - T) * D_MODEL;
    int pos_len = 2 * T - 1;

    // 4. Conformer blocks
    for (int blk = 0; blk < N_BLOCKS; blk++) {
        auto& b = w->blocks[blk];

        // --- FF1 (half-step residual) ---
        layer_norm_fp16(x, b.ff1_ln_w, b.ff1_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        // FF1 linear1 + SiLU: [T,1024] × [1024,4096] → [T,4096]
        gnn(ln_out, T, D_MODEL, b.ff1_w1, D_FF, ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);

        gnn(ff_mid, T, D_FF, b.ff1_w2, D_MODEL, ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.mhsa_ln_w, b.mhsa_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        {
            // Fused QKV projection: [T, D] × [D, 3D] → [T, 3D]
            gnn(ln_out, T, D_MODEL, qkv_w[blk], 3 * D_MODEL, qkv);

            // Fused split+transpose+bias: [T, 3D] → q_u, q_v (with bias), K, V
            half* K_h = k;
            half* V_h = v;
            split_transpose_qkv_bias_fp16(qkv, b.pos_bias_u, b.pos_bias_v,
                                           q_u, q_v_buf, K_h, V_h,
                                           T, N_HEADS, HEAD_DIM, stream);

            // Position encoding projection
            half* pos_temp = pos_proj;
            gnn(pos_enc_T, pos_len, D_MODEL, b.pos_w, D_MODEL, pos_temp);

            // Position scores via strided batched GEMM
            // Row-major NT: pos_scores[h,T,pos_len] = q_v[h,T,HEAD_DIM] @ pos[h,pos_len,HEAD_DIM]^T
            // pos_temp has non-standard ld=D_MODEL (interleaved heads), stride=HEAD_DIM between heads
            batched_gemm_nt_ex(stream,
                q_v_buf, HEAD_DIM, (long long)T * HEAD_DIM,        // A: [T, HEAD_DIM] per head
                pos_temp, D_MODEL, (long long)HEAD_DIM,            // B: [pos_len, HEAD_DIM] ld=D_MODEL
                pos_scores, pos_len, (long long)T * pos_len,       // C: [T, pos_len] per head
                N_HEADS, T, pos_len, HEAD_DIM);

            float scale = 1.0f / sqrtf((float)HEAD_DIM);

            // Content attention scores: [8, T, T] = q_u @ K^T
            batched_gemm_nt(stream, q_u, K_h, scores,
                            N_HEADS, T, T, HEAD_DIM,
                            (long long)T * HEAD_DIM, (long long)T * HEAD_DIM, (long long)T * T);
            // Fused: (content + pos_skew) * scale + softmax
            fused_score_softmax_fp16(scores, pos_scores, scores,
                                      N_HEADS, T, scale, stream);
            // Weighted sum: [8, T, 128] = scores @ V
            batched_gemm_nn(stream, scores, V_h, attn_out,
                            N_HEADS, T, HEAD_DIM, T,
                            (long long)T * T, (long long)T * HEAD_DIM, (long long)T * HEAD_DIM);
            // Transpose [8, T, 128] → [T, 8, 128] = [T, 1024]
            transpose_0213_fp16(attn_out, ff_out, N_HEADS, T, HEAD_DIM, stream);

            // Output projection
            gnn(ff_out, T, D_MODEL, b.out_w, D_MODEL, mhsa_out);
        }

        // Fused: x += mhsa_out, then ln_out = LN(x)
        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.conv_ln_w, b.conv_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        // Pointwise conv1 + GLU
        gnt(ln_out, T, D_MODEL, b.conv_pw1_w, D_CONV_PW, conv_mid);
        glu_fp16(conv_mid, conv_glu, T, D_MODEL, stream);

        // Fused depthwise conv 1D k=9 + SiLU
        depthwise_conv1d_k9_silu_fp16(conv_glu, b.conv_dw_w, b.conv_dw_b,
                                       conv_dw, T, D_MODEL, stream);

        // Pointwise conv2
        gnt(conv_dw, T, D_MODEL, b.conv_pw2_w, D_MODEL, mhsa_out);

        // Fused: x += conv_out, then ln_out = LN(x)
        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.ff2_ln_w, b.ff2_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        // FF2 linear1 + SiLU
        gnn(ln_out, T, D_MODEL, b.ff2_w1, D_FF, ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);

        gnn(ff_mid, T, D_FF, b.ff2_w2, D_MODEL, ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.final_ln_w, b.final_ln_b, x, T, D_MODEL, 1e-5f, stream);
    }

    // x now holds encoder output [T, D_MODEL] in FP16

    // Precompute encoder projections for all T frames (used by fused joint kernel)
    gnn_bias(x, T, D_MODEL, w->enc_proj_w, D_JOINT, w->enc_proj_b, enc_proj_all);

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
    auto gnn_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nn_bias(stream, X, m, k, W, n, bias, Y);
    };
    auto gnt_b = [&](const half* X, int m, int k, const half* W, int n, const half* bias, half* Y) {
        gemm_nt_bias(stream, X, m, k, W, n, bias, Y);
    };

    // 1. Embed + concat with h[0] for LSTM0 input: lstm_input = [embed; h[0]]
    embed_concat_fp16(w->embed_w, prev_token, lstm_h[0], lstm_input, D_PRED, stream);

    // 2. LSTM layer 0 — single cuBLAS call with combined [W_ih|W_hh] weights
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[0], 4 * D_PRED, lstm_combined_bias[0], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[0], lstm_h_out[0], lstm_c_out[0], D_PRED, stream);

    // 3. Concat h_out[0] with h[1] for LSTM1 input: lstm_input = [h_out[0]; h[1]]
    concat_vectors_fp16(lstm_h_out[0], lstm_h[1], lstm_input, D_PRED, stream);

    // 4. LSTM layer 1 — single cuBLAS call with combined weights
    gnt_b(lstm_input, 1, 2 * D_PRED, lstm_combined_w[1], 4 * D_PRED, lstm_combined_bias[1], lstm_gates);
    lstm_cell_fp16(lstm_gates, lstm_c[1], lstm_h_out[1], lstm_c_out[1], D_PRED, stream);

    // 5. Joint network — cuBLAS for dec_proj + out_proj (multi-SM beats single-SM for 2MB weights)
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
