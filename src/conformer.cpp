// conformer.cpp — Weight loading for CUDA backend (Phase 1)
//
// Loads a flat binary weight file (produced by scripts/export_weights.py)
// into a single contiguous GPU allocation, then assigns struct field pointers
// by matching tensor names from the file header.

#include "conformer.h"
#include "kernels.h"

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

#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

// cuDNN SDPA tensor UIDs
enum SdpaUid : int64_t {
    SDPA_Q = 1, SDPA_K = 2, SDPA_V = 3,
    SDPA_O = 4, SDPA_STATS = 5, SDPA_BIAS = 6
};

// ---------------------------------------------------------------------------
// CUDA error checking (same macro as parakeet.cpp)
// ---------------------------------------------------------------------------

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
#endif

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
// Weights::load
// ---------------------------------------------------------------------------

Weights Weights::load(const std::string& path, cudaStream_t stream) {
    Weights w;

    // mmap the file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open weights: %s\n", path.c_str());
        std::exit(1);
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", path.c_str());
        close(fd);
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
        close(fd);
        std::exit(1);
    }

    uint32_t version;
    memcpy(&version, base + 4, 4);
    if (version != PRKT_VERSION) {
        fprintf(stderr, "Unsupported weight file version %u (expected %u)\n", version, PRKT_VERSION);
        munmap(mapped, file_size);
        close(fd);
        std::exit(1);
    }

    uint64_t header_len;
    memcpy(&header_len, base + 8, 8);

    // Parse text index
    const char* header_text = (const char*)(base + 16);
    w.tensors = parse_header(header_text, header_len);

    // Build name lookup
    for (size_t i = 0; i < w.tensors.size(); i++) {
        w.name_to_idx[w.tensors[i].name] = i;
    }

    // Compute data section start
    size_t header_end = 16 + header_len;
    size_t data_start = align_up(header_end, HEADER_ALIGN);

    // Compute total GPU allocation needed
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

    // Allocate GPU memory and copy data in one shot
    CUDA_CHECK(cudaMalloc(&w.gpu_data, total_data));
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(w.gpu_data, base + data_start, total_data,
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(w.gpu_data, base + data_start, total_data,
                               cudaMemcpyHostToDevice));
    }

    // Clean up mmap
    munmap(mapped, file_size);
    close(fd);

    // Assign struct field pointers by matching tensor names.
    uint8_t* gpu_base = (uint8_t*)w.gpu_data;

    auto lookup = [&](const std::string& name) -> half* {
        auto it = w.name_to_idx.find(name);
        if (it == w.name_to_idx.end()) return nullptr;
        return (half*)(gpu_base + w.tensors[it->second].offset);
    };

    // ---------------------------------------------------------------------------
    // Subsampling (pre_encode)
    // Conv layers: 0, 2, 3, 5, 6
    // ---------------------------------------------------------------------------
    for (int i : {0, 2, 3, 5, 6}) {
        std::string pre = "encoder/pre_encode.conv." + std::to_string(i);
        w.sub_conv[i].weight = lookup(pre + ".weight");
        w.sub_conv[i].bias   = lookup(pre + ".bias");
    }
    w.sub_out_w = lookup("encoder/pre_encode.out.weight");
    w.sub_out_b = lookup("encoder/pre_encode.out.bias");

    // ---------------------------------------------------------------------------
    // Conformer blocks (x24)
    // ---------------------------------------------------------------------------
    for (int i = 0; i < 24; i++) {
        auto& blk = w.blocks[i];
        std::string pre = "encoder/layers." + std::to_string(i);

        // FF1
        blk.ff1_ln_w = lookup(pre + ".norm_feed_forward1.weight");
        blk.ff1_ln_b = lookup(pre + ".norm_feed_forward1.bias");
        blk.ff1_w1   = lookup(pre + ".feed_forward1.linear1.weight");
        blk.ff1_w2   = lookup(pre + ".feed_forward1.linear2.weight");

        // MHSA
        blk.mhsa_ln_w = lookup(pre + ".norm_self_att.weight");
        blk.mhsa_ln_b = lookup(pre + ".norm_self_att.bias");
        blk.q_w       = lookup(pre + ".self_attn.linear_q.weight");
        blk.k_w       = lookup(pre + ".self_attn.linear_k.weight");
        blk.v_w       = lookup(pre + ".self_attn.linear_v.weight");
        blk.pos_w     = lookup(pre + ".self_attn.linear_pos.weight");
        blk.pos_bias_u = lookup(pre + ".self_attn.pos_bias_u");
        blk.pos_bias_v = lookup(pre + ".self_attn.pos_bias_v");
        blk.out_w     = lookup(pre + ".self_attn.linear_out.weight");

        // Conv module
        blk.conv_ln_w  = lookup(pre + ".norm_conv.weight");
        blk.conv_ln_b  = lookup(pre + ".norm_conv.bias");
        blk.conv_pw1_w = lookup(pre + ".conv.pointwise_conv1.weight");
        blk.conv_dw_w  = lookup(pre + ".conv.depthwise_conv.weight");
        blk.conv_dw_b  = lookup(pre + ".conv.depthwise_conv.bias");
        blk.conv_pw2_w = lookup(pre + ".conv.pointwise_conv2.weight");

        // FF2
        blk.ff2_ln_w = lookup(pre + ".norm_feed_forward2.weight");
        blk.ff2_ln_b = lookup(pre + ".norm_feed_forward2.bias");
        blk.ff2_w1   = lookup(pre + ".feed_forward2.linear1.weight");
        blk.ff2_w2   = lookup(pre + ".feed_forward2.linear2.weight");

        // Final LN
        blk.final_ln_w = lookup(pre + ".norm_out.weight");
        blk.final_ln_b = lookup(pre + ".norm_out.bias");
    }

    // ---------------------------------------------------------------------------
    // Decoder: embedding + LSTM (ONNX format)
    // ---------------------------------------------------------------------------
    w.embed_w = lookup("decoder/decoder.prediction.embed.weight");

    // LSTM layer 0
    w.lstm0_w_ih = lookup("decoder/decoder.dec_rnn.lstm.weight_ih");
    w.lstm0_w_hh = lookup("decoder/decoder.dec_rnn.lstm.weight_hh");
    w.lstm0_bias = lookup("decoder/decoder.dec_rnn.lstm.bias");

    // LSTM layer 1
    w.lstm1_w_ih = lookup("decoder/decoder.dec_rnn.lstm.1.weight_ih");
    w.lstm1_w_hh = lookup("decoder/decoder.dec_rnn.lstm.1.weight_hh");
    w.lstm1_bias = lookup("decoder/decoder.dec_rnn.lstm.1.bias");

    // ---------------------------------------------------------------------------
    // Joint network
    // ---------------------------------------------------------------------------
    w.enc_proj_w = lookup("decoder/joint.enc.weight");
    w.enc_proj_b = lookup("decoder/joint.enc.bias");
    w.dec_proj_w = lookup("decoder/joint.pred.weight");
    w.dec_proj_b = lookup("decoder/joint.pred.bias");
    w.out_proj_w = lookup("decoder/joint.joint_net.joint_net.2.weight");
    w.out_proj_b = lookup("decoder/joint.joint_net.2.bias");

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

// Helper: allocate FP16 buffer
static half* alloc_fp16(size_t count) {
    half* p;
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(half)));
    return p;
}

// ---------------------------------------------------------------------------
// cuDNN SDPA graph builder + cache (keyed by T)
// ---------------------------------------------------------------------------

struct SdpaGraph {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
};

static std::unordered_map<int, SdpaGraph> sdpa_cache;

static SdpaGraph& get_sdpa_graph(cudnnHandle_t cudnn, int T) {
    auto it = sdpa_cache.find(T);
    if (it != sdpa_cache.end()) return it->second;

    int64_t b = 1, h = N_HEADS, s = T, d = HEAD_DIM;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V: [1, H, T, D] with H-major layout [H, T, D] contiguous
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q").set_uid(SDPA_Q)
        .set_dim({b, h, s, d})
        .set_stride({h*s*d, s*d, d, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K").set_uid(SDPA_K)
        .set_dim({b, h, s, d})
        .set_stride({h*s*d, s*d, d, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V").set_uid(SDPA_V)
        .set_dim({b, h, s, d})
        .set_stride({h*s*d, s*d, d, 1}));

    // Additive bias: [1, H, T, T] — per-head relative position bias
    auto Bias = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Bias").set_uid(SDPA_BIAS)
        .set_dim({b, h, s, s})
        .set_stride({h*s*s, s*s, s, 1}));

    auto sdpa_opts = fe::graph::SDPA_attributes()
        .set_name("flash_attention")
        .set_attn_scale(scale)
        .set_bias(Bias)
        .set_generate_stats(false);

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_opts);

    // Output: [1, H, T, D] with S-major layout [T, H, D] contiguous
    // This lets us skip the transpose_0213 after attention
    O->set_output(true)
      .set_dim({b, h, s, d})
      .set_stride({s*h*d, d, h*d, 1})
      .set_uid(SDPA_O);

    auto status = graph->build(cudnn, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA graph build failed for T=%d: %s\n",
                T, status.get_message().c_str());
        // Return an entry with null graph to signal failure
        auto& entry = sdpa_cache[T];
        entry.graph = nullptr;
        return entry;
    }

    auto& entry = sdpa_cache[T];
    entry.graph = graph;
    graph->get_workspace_size(entry.workspace_size);
    return entry;
}

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

    // Subsampling buffers (max intermediate size: after conv.0 = [256, 64, T/2])
    int max_sub = SUB_CHANNELS * (max_mel_frames / 2 + 1) * 65;  // generous
    sub_buf[0] = alloc_fp16(max_sub);
    sub_buf[1] = alloc_fp16(max_sub);
    CUDA_CHECK(cudaMalloc(&mel_fp32, 128 * max_mel_frames * sizeof(float)));
    mel_fp16 = alloc_fp16(128 * max_mel_frames);

    // Encoder buffers
    x        = alloc_fp16(T_max * D_MODEL);
    ln_out   = alloc_fp16(T_max * D_MODEL);
    ff_mid   = alloc_fp16(T_max * D_FF);
    ff_out   = alloc_fp16(T_max * D_MODEL);
    qkv      = alloc_fp16(T_max * 3 * D_MODEL);
    q        = alloc_fp16(N_HEADS * T_max * HEAD_DIM);
    k        = alloc_fp16(N_HEADS * T_max * HEAD_DIM);
    v        = alloc_fp16(N_HEADS * T_max * HEAD_DIM);

    // Pre-concatenate QKV weights: [D_MODEL, 3*D_MODEL] per block
    // Q_w, K_w, V_w are each [D_MODEL, D_MODEL] (ONNX MatMul: Y = X @ W)
    // Concatenate along output dim: W_qkv[i, j] for j<D = Q_w[i,j], j<2D = K_w[i,j-D], etc.
    for (int b = 0; b < N_BLOCKS; b++) {
        CUDA_CHECK(cudaMalloc(&qkv_w[b], D_MODEL * 3 * D_MODEL * sizeof(half)));
        // Q_w, K_w, V_w are [D_MODEL, D_MODEL] = [1024, 1024] in row-major
        // W_qkv should be [D_MODEL, 3*D_MODEL] = [1024, 3072]
        // Row i of W_qkv = [row_i_of_Q | row_i_of_K | row_i_of_V]
        for (int row = 0; row < D_MODEL; row++) {
            CUDA_CHECK(cudaMemcpyAsync(
                qkv_w[b] + row * 3 * D_MODEL,
                weights.blocks[b].q_w + row * D_MODEL,
                D_MODEL * sizeof(half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                qkv_w[b] + row * 3 * D_MODEL + D_MODEL,
                weights.blocks[b].k_w + row * D_MODEL,
                D_MODEL * sizeof(half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                qkv_w[b] + row * 3 * D_MODEL + 2 * D_MODEL,
                weights.blocks[b].v_w + row * D_MODEL,
                D_MODEL * sizeof(half), cudaMemcpyDeviceToDevice, stream));
        }
    }
    pos_enc  = alloc_fp16((2 * T_max) * D_MODEL);
    pos_proj = alloc_fp16((2 * T_max) * D_MODEL);

    // Pre-compute position encoding for T_max (reused via offset for any T <= T_max)
    generate_pos_encoding_gpu(pos_enc, T_max, D_MODEL, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    q_u      = alloc_fp16(N_HEADS * T_max * HEAD_DIM);
    q_v_buf  = alloc_fp16(N_HEADS * T_max * HEAD_DIM);
    scores   = alloc_fp16(N_HEADS * T_max * T_max);
    pos_scores = alloc_fp16(N_HEADS * T_max * (2 * T_max));
    attn_out = alloc_fp16(N_HEADS * T_max * HEAD_DIM);
    mhsa_out = alloc_fp16(T_max * D_MODEL);
    conv_mid = alloc_fp16(T_max * D_CONV_PW);
    conv_glu = alloc_fp16(T_max * D_MODEL);
    conv_dw  = alloc_fp16(T_max * D_MODEL);
    pos_bias = alloc_fp16(N_HEADS * T_max * T_max);

    // cuDNN flash attention
    auto cudnn_status = cudnnCreate(&cudnn);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cudnnCreate failed: %s\n", cudnnGetErrorString(cudnn_status));
        cudnn = nullptr;
    } else {
        cudnnSetStream(cudnn, stream);
        // Pre-build SDPA graph for warmup T (actual T built lazily)
        sdpa_workspace_size = 32 * 1024 * 1024;  // 32MB initial
        CUDA_CHECK(cudaMalloc(&sdpa_workspace, sdpa_workspace_size));
        CUDA_CHECK(cudaMalloc(&sdpa_stats, N_HEADS * T_max * sizeof(float)));
    }

    // Decoder buffers
    dec_embed  = alloc_fp16(D_PRED);
    lstm_gates = alloc_fp16(4 * D_PRED);
    lstm_h[0]     = alloc_fp16(D_PRED);
    lstm_h[1]     = alloc_fp16(D_PRED);
    lstm_c[0]     = alloc_fp16(D_PRED);
    lstm_c[1]     = alloc_fp16(D_PRED);
    lstm_h_out[0] = alloc_fp16(D_PRED);
    lstm_h_out[1] = alloc_fp16(D_PRED);
    lstm_c_out[0] = alloc_fp16(D_PRED);
    lstm_c_out[1] = alloc_fp16(D_PRED);
    enc_proj   = alloc_fp16(D_JOINT);
    dec_proj   = alloc_fp16(D_JOINT);
    joint_act  = alloc_fp16(D_JOINT);
    joint_out  = alloc_fp16(D_OUTPUT);
}

// ---------------------------------------------------------------------------
// cublasLt GEMM algorithm cache (forward declaration for free())
// ---------------------------------------------------------------------------

struct GemmKey {
    int m, n, k;
    cublasOperation_t opA, opB;
    bool accumulate;
    bool operator==(const GemmKey& o) const {
        return m == o.m && n == o.n && k == o.k && opA == o.opA && opB == o.opB && accumulate == o.accumulate;
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
    if (lt_workspace) { cudaFree(lt_workspace); lt_workspace = nullptr; }
    for (auto& [t, gp] : graph_cache) {
        cudaGraphExecDestroy(gp.second);
        cudaGraphDestroy(gp.first);
    }
    graph_cache.clear();
    if (mel_fp32) { cudaFree(mel_fp32); mel_fp32 = nullptr; }
    auto f = [](half*& p) { if (p) { cudaFree(p); p = nullptr; } };
    f(mel_fp16); f(sub_buf[0]); f(sub_buf[1]);
    f(x); f(ln_out); f(ff_mid); f(ff_out);
    f(qkv); f(q); f(k); f(v); f(pos_enc); f(pos_proj);
    for (int b = 0; b < N_BLOCKS; b++) {
        if (qkv_w[b]) { cudaFree(qkv_w[b]); qkv_w[b] = nullptr; }
    }
    f(q_u); f(q_v_buf); f(scores); f(pos_scores); f(attn_out); f(mhsa_out);
    f(conv_mid); f(conv_glu); f(conv_dw); f(pos_bias);
    if (sdpa_stats) { cudaFree(sdpa_stats); sdpa_stats = nullptr; }
    if (sdpa_workspace) { cudaFree(sdpa_workspace); sdpa_workspace = nullptr; }
    sdpa_cache.clear();
    if (cudnn) { cudnnDestroy(cudnn); cudnn = nullptr; }
    f(dec_embed); f(lstm_gates);
    f(lstm_h[0]); f(lstm_h[1]); f(lstm_c[0]); f(lstm_c[1]);
    f(lstm_h_out[0]); f(lstm_h_out[1]); f(lstm_c_out[0]); f(lstm_c_out[1]);
    f(enc_proj); f(dec_proj); f(joint_act); f(joint_out);
}

// ---------------------------------------------------------------------------
// cublasLt GEMM with algorithm caching
// ---------------------------------------------------------------------------

static GemmPlan& get_gemm_plan(cublasLtHandle_t lt, void* workspace, size_t ws_size,
                                int col_m, int col_n, int col_k,
                                int ldA, int ldB, int ldC,
                                cublasOperation_t opA, cublasOperation_t opB,
                                bool accumulate) {
    GemmKey key{col_m, col_n, col_k, opA, opB, accumulate};
    auto it = gemm_cache.find(key);
    if (it != gemm_cache.end()) return it->second;

    GemmPlan plan;
    plan.valid = false;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.matmul_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

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
                      bool accumulate) {
    half alpha = __float2half(1.0f);
    half beta = __float2half(accumulate ? 1.0f : 0.0f);

    GemmPlan& plan = get_gemm_plan(lt, workspace, ws_size,
                                     col_m, col_n, col_k, ldA, ldB, ldC,
                                     opA, opB, accumulate);
    if (plan.valid) {
        CUBLAS_CHECK(cublasLtMatmul(lt, plan.matmul_desc,
            &alpha, A, plan.layout_A, B, plan.layout_B,
            &beta, C, plan.layout_C, C, plan.layout_C,
            &plan.algo, workspace, ws_size, stream));
    } else {
        // Fallback to cublasGemmEx
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
    // Local GEMM helpers that capture cublasLt context
    auto gnn = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nn(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, Y);
    };
    auto gnt = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nt(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, Y);
    };

    // 1. Upload and cast mel to FP16, then transpose
    //    mel_fp32_host is [128, T_mel] on host
    //    ONNX model transposes to [T_mel, 128] before conv2d
    CUDA_CHECK(cudaMemcpyAsync(mel_fp32, mel_fp32_host, 128 * T_mel * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    cast_fp32_to_fp16(mel_fp32, mel_fp16, 128 * T_mel, stream);

    // Transpose mel from [128, T_mel] to [T_mel, 128]
    transpose_fp16(mel_fp16, sub_buf[0], 128, T_mel, stream);
    // sub_buf[0] is now [T_mel, 128] which acts as [C_in=1, H=T_mel, W=128] for conv2d

    // 2. Subsampling: [1, T_mel, 128] → [T', 1024]
    //    conv.0 (1→256, 3x3, s=2) → ReLU
    int H = T_mel, W = 128;
    conv2d_fp16(sub_buf[0], w->sub_conv[0].weight, w->sub_conv[0].bias,
                sub_buf[1], 1, H, W, SUB_CHANNELS, 3, 3, 2, 1, 1, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;   // 64
    relu_inplace_fp16(sub_buf[1], SUB_CHANNELS * H * W, stream);

    //    conv.2 (depthwise 256, 3x3, s=2) → conv.3 (pointwise 256→256, 1x1) → ReLU
    conv2d_fp16(sub_buf[1], w->sub_conv[2].weight, w->sub_conv[2].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;
    W = (W + 2 * 1 - 3) / 2 + 1;   // 32
    conv2d_fp16(sub_buf[0], w->sub_conv[3].weight, w->sub_conv[3].bias,
                sub_buf[1], SUB_CHANNELS, H, W, SUB_CHANNELS, 1, 1, 1, 0, 1, stream);
    relu_inplace_fp16(sub_buf[1], SUB_CHANNELS * H * W, stream);

    //    conv.5 (depthwise 256, 3x3, s=2) → conv.6 (pointwise 256→256, 1x1) → ReLU
    conv2d_fp16(sub_buf[1], w->sub_conv[5].weight, w->sub_conv[5].bias,
                sub_buf[0], SUB_CHANNELS, H, W, SUB_CHANNELS, 3, 3, 2, 1, SUB_CHANNELS, stream);
    H = (H + 2 * 1 - 3) / 2 + 1;   // T' (encoder frames)
    W = (W + 2 * 1 - 3) / 2 + 1;   // 16
    conv2d_fp16(sub_buf[0], w->sub_conv[6].weight, w->sub_conv[6].bias,
                sub_buf[1], SUB_CHANNELS, H, W, SUB_CHANNELS, 1, 1, 1, 0, 1, stream);
    relu_inplace_fp16(sub_buf[1], SUB_CHANNELS * H * W, stream);

    // Reshape [256, H, 16] → [H, 256*16] = [T', 4096] then linear → [T', 1024]
    // This is [C, H, W] → [H, C*W] (permute(1,0,2) + flatten)
    int T = H;  // encoder frames = time dim after 3x stride-2
    reshape_chw_to_hcw_fp16(sub_buf[1], sub_buf[0], SUB_CHANNELS, T, W, stream);
    // sub_buf[0] is now [T, 4096]
    gnn(sub_buf[0], T, SUB_CHANNELS * W, w->sub_out_w, D_MODEL, x);
    // Add bias
    if (w->sub_out_b)
        bias_add_inplace_fp16(x, w->sub_out_b, T, D_MODEL, stream);

    // 3. Position encoding: slice from pre-computed T_max encoding
    //    pos_enc for T is at offset (T_max - T) * D_MODEL in the T_max encoding
    half* pos_enc_T = pos_enc + (T_max - T) * D_MODEL;
    int pos_len = 2 * T - 1;

    // 4. Conformer blocks
    for (int blk = 0; blk < N_BLOCKS; blk++) {
        auto& b = w->blocks[blk];

        // --- FF1 (half-step residual) ---
        layer_norm_fp16(x, b.ff1_ln_w, b.ff1_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);
        gnn(ln_out, T, D_MODEL, b.ff1_w1, D_FF, ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);
        gnn(ff_mid, T, D_FF, b.ff1_w2, D_MODEL, ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.mhsa_ln_w, b.mhsa_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        {
            // Fused QKV projection: [T, D] × [D, 3D] → [T, 3D]
            gnn(ln_out, T, D_MODEL, qkv_w[blk], 3 * D_MODEL, qkv);

            // Split + transpose: [T, 3*D] → 3× [8, T, 128]
            half* Q_h = q;
            half* K_h = k;
            half* V_h = v;
            split_transpose_3way_fp16(qkv, Q_h, K_h, V_h, T, N_HEADS, HEAD_DIM, stream);

            // Position encoding projection → transpose
            half* pos_temp = ff_out;
            gnn(pos_enc_T, pos_len, D_MODEL, b.pos_w, D_MODEL, pos_temp);
            transpose_0213_fp16(pos_temp, pos_proj, pos_len, N_HEADS, HEAD_DIM, stream);

            // q_u = Q + pos_bias_u, q_v = Q + pos_bias_v  (fused: 2→1 kernel)
            add_pos_bias_dual_fp16(Q_h, b.pos_bias_u, b.pos_bias_v,
                                    q_u, q_v_buf, N_HEADS, T, HEAD_DIM, stream);

            // Position scores: [8, T, pos_len] = q_v[8, T, 128] @ pos[8, pos_len, 128]^T
            batched_gemm_nt(cublas, q_v_buf, pos_proj, pos_scores,
                            N_HEADS, T, pos_len, HEAD_DIM,
                            (long long)T * HEAD_DIM, (long long)pos_len * HEAD_DIM,
                            (long long)T * pos_len);

            // Skew position scores: [8, T, 2T-1] → [8, T, T]
            // Pre-scale by 1/sqrt(d) so cuDNN SDPA's (Q×K^T×scale + bias) matches
            // our original (Q×K^T + bias)×scale
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            rel_pos_skew_scale_fp16(pos_scores, pos_bias, N_HEADS, T, scale, stream);

            // cuDNN Flash Attention: fuses content_scores + add_bias + scale + softmax + weighted_sum
            // Input Q=q_u, K, V in [H, T, D] layout; Bias=pos_bias in [H, T, T]
            // Output directly in [T, H, D] layout (no transpose needed)
            SdpaGraph& sg = get_sdpa_graph(cudnn, T);
            if (sg.graph) {
                // Ensure workspace is large enough
                if (sg.workspace_size > (int64_t)sdpa_workspace_size) {
                    cudaFree(sdpa_workspace);
                    sdpa_workspace_size = sg.workspace_size;
                    CUDA_CHECK(cudaMalloc(&sdpa_workspace, sdpa_workspace_size));
                }

                std::unordered_map<int64_t, void*> variant_pack = {
                    {SDPA_Q,    (void*)q_u},
                    {SDPA_K,    (void*)K_h},
                    {SDPA_V,    (void*)V_h},
                    {SDPA_BIAS, (void*)pos_bias},
                    {SDPA_O,    (void*)ff_out},  // output in [T, H, D] layout
                };
                auto status = sg.graph->execute(cudnn, variant_pack, sdpa_workspace);
                if (!status.is_good()) {
                    fprintf(stderr, "cuDNN SDPA execute failed (T=%d): %s\n",
                            T, status.get_message().c_str());
                }
            } else {
                // Fallback: original path without flash attention
                batched_gemm_nt(cublas, q_u, K_h, scores,
                                N_HEADS, T, T, HEAD_DIM,
                                (long long)T * HEAD_DIM, (long long)T * HEAD_DIM, (long long)T * T);
                float scale = 1.0f / sqrtf((float)HEAD_DIM);
                fused_score_softmax_fp16(scores, pos_scores, scores,
                                          N_HEADS, T, scale, stream);
                batched_gemm_nn(cublas, scores, V_h, attn_out,
                                N_HEADS, T, HEAD_DIM, T,
                                (long long)T * T, (long long)T * HEAD_DIM, (long long)T * HEAD_DIM);
                transpose_0213_fp16(attn_out, ff_out, N_HEADS, T, HEAD_DIM, stream);
            }

            // Output projection (ff_out is now [T, 1024] in both paths)
            gnn(ff_out, T, D_MODEL, b.out_w, D_MODEL, mhsa_out);
        }

        // Fused: x += mhsa_out, then ln_out = LN(x)
        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.conv_ln_w, b.conv_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);

        // Pointwise conv1: [T, 1024] → [T, 2048]
        gnt(ln_out, T, D_MODEL, b.conv_pw1_w, D_CONV_PW, conv_mid);

        // GLU: [T, 2048] → [T, 1024]
        glu_fp16(conv_mid, conv_glu, T, D_MODEL, stream);

        // Fused depthwise conv 1D k=9 + SiLU: [T, 1024] → [T, 1024]
        depthwise_conv1d_k9_silu_fp16(conv_glu, b.conv_dw_w, b.conv_dw_b,
                                       conv_dw, T, D_MODEL, stream);

        // Pointwise conv2: [T, 1024] → [T, 1024]
        gnt(conv_dw, T, D_MODEL, b.conv_pw2_w, D_MODEL, mhsa_out);

        // Fused: x += conv_out, then ln_out = LN(x)
        residual_add_layer_norm_fp16(x, mhsa_out, 1.0f,
            b.ff2_ln_w, b.ff2_ln_b, ln_out, T, D_MODEL, 1e-5f, stream);
        gnn(ln_out, T, D_MODEL, b.ff2_w1, D_FF, ff_mid);
        silu_inplace_fp16(ff_mid, T * D_FF, stream);
        gnn(ff_mid, T, D_FF, b.ff2_w2, D_MODEL, ff_out);
        residual_add_layer_norm_fp16(x, ff_out, 0.5f,
            b.final_ln_w, b.final_ln_b, x, T, D_MODEL, 1e-5f, stream);
    }

    // x now holds encoder output [T, D_MODEL] in FP16
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
    // Local GEMM helpers that capture cublasLt context
    auto gnn = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nn(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, Y);
    };
    auto gnt = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nt(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, Y);
    };
    auto gnt_acc = [&](const half* X, int m, int k, const half* W, int n, half* Y) {
        gemm_nt_accum(cublaslt, cublas, lt_workspace, lt_workspace_size, stream, X, m, k, W, n, Y);
    };

    // 1. Embedding gather
    embedding_gather_fp16(w->embed_w, prev_token, dec_embed, D_PRED, stream);

    // 2. LSTM layer 0
    gnt(dec_embed, 1, D_PRED, w->lstm0_w_ih, 4 * D_PRED, lstm_gates);
    gnt_acc(lstm_h[0], 1, D_PRED, w->lstm0_w_hh, 4 * D_PRED, lstm_gates);
    bias_add_inplace_fp16(lstm_gates, w->lstm0_bias, 1, 4 * D_PRED, stream);
    bias_add_inplace_fp16(lstm_gates, w->lstm0_bias + 4 * D_PRED, 1, 4 * D_PRED, stream);
    lstm_cell_fp16(lstm_gates, lstm_c[0], lstm_h_out[0], lstm_c_out[0], D_PRED, stream);

    // 3. LSTM layer 1
    gnt(lstm_h_out[0], 1, D_PRED, w->lstm1_w_ih, 4 * D_PRED, lstm_gates);
    gnt_acc(lstm_h[1], 1, D_PRED, w->lstm1_w_hh, 4 * D_PRED, lstm_gates);
    bias_add_inplace_fp16(lstm_gates, w->lstm1_bias, 1, 4 * D_PRED, stream);
    bias_add_inplace_fp16(lstm_gates, w->lstm1_bias + 4 * D_PRED, 1, 4 * D_PRED, stream);
    lstm_cell_fp16(lstm_gates, lstm_c[1], lstm_h_out[1], lstm_c_out[1], D_PRED, stream);

    // 4. Joint network
    half* enc_frame = x + enc_frame_idx * D_MODEL;
    gnn(enc_frame, 1, D_MODEL, w->enc_proj_w, D_JOINT, enc_proj);
    if (w->enc_proj_b)
        bias_add_inplace_fp16(enc_proj, w->enc_proj_b, 1, D_JOINT, stream);

    gnn(lstm_h_out[1], 1, D_PRED, w->dec_proj_w, D_JOINT, dec_proj);
    if (w->dec_proj_b)
        bias_add_inplace_fp16(dec_proj, w->dec_proj_b, 1, D_JOINT, stream);

    add_relu_fp16(enc_proj, dec_proj, joint_act, D_JOINT, stream);

    gnn(joint_act, 1, D_JOINT, w->out_proj_w, D_OUTPUT, joint_out);
    if (w->out_proj_b)
        bias_add_inplace_fp16(joint_out, w->out_proj_b, 1, D_OUTPUT, stream);

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
