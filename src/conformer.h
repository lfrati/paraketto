// conformer.h — Weight loading + CUDA inference for Parakeet conformer
//
// FP8 E4M3 quantized weights with CUTLASS FP8 GEMMs on SM120 (Blackwell).
//
// Defines:
//   Weights   — pointers into GPU weight allocation (loaded from weights.bin)
//   CudaModel — pre-allocated buffers for forward passes

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ---------------------------------------------------------------------------
// Weight file format constants
// ---------------------------------------------------------------------------

static constexpr uint32_t PRKT_MAGIC   = 0x544B5250;  // "PRKT" little-endian
static constexpr uint32_t PRKT_VERSION = 1;
static constexpr size_t   HEADER_ALIGN = 4096;

// ---------------------------------------------------------------------------
// Tensor descriptor (parsed from header)
// ---------------------------------------------------------------------------

struct TensorDesc {
    std::string name;
    size_t offset;       // offset from data start
    size_t size_bytes;
    std::string dtype;
    std::vector<int> shape;
};

// ---------------------------------------------------------------------------
// Weights struct — all model weights on GPU
// ---------------------------------------------------------------------------

struct Weights {
    void* gpu_data = nullptr;        // single contiguous GPU allocation
    size_t gpu_data_size = 0;

    // Prefetch state (temporary — cleared after upload)
    void* mmap_ptr = nullptr;
    size_t mmap_size = 0;
    size_t data_offset = 0;          // offset to data section within mmap

    // Parsed index (kept for diagnostics)
    std::vector<TensorDesc> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;

    // ---------------------------------------------------------------------------
    // Subsampling (pre_encode): 5 conv layers + 1 linear projection
    //   conv.0: depthwise [256, 1, 3, 3] stride=2
    //   conv.2: depthwise [256, 1, 3, 3] stride=2
    //   conv.3: pointwise [256, 256, 1, 1]
    //   conv.5: depthwise [256, 1, 3, 3] stride=2
    //   conv.6: pointwise [256, 256, 1, 1]
    //   out:    linear    [4096, 1024] (flattened 256*16 -> 1024)
    // ---------------------------------------------------------------------------
    struct SubConv {
        half *weight = nullptr, *bias = nullptr;
    };
    SubConv sub_conv[7];  // indexed by conv layer number (0,2,3,5,6 used; 1,4 unused)
    half *sub_out_w = nullptr, *sub_out_b = nullptr;  // pre_encode.out

    // ---------------------------------------------------------------------------
    // Conformer blocks (x24)
    //
    // Per block tensors (from ONNX inspection):
    //   norm_feed_forward1.{weight,bias}     LayerNorm
    //   feed_forward1.linear1.weight         [1024, 4096] (no bias)
    //   feed_forward1.linear2.weight         [4096, 1024]
    //   norm_self_att.{weight,bias}          LayerNorm
    //   self_attn.linear_q.weight            [1024, 1024]
    //   self_attn.linear_k.weight            [1024, 1024]
    //   self_attn.linear_v.weight            [1024, 1024]
    //   self_attn.linear_pos.weight          [1024, 1024]
    //   self_attn.pos_bias_u                 [8, 128]
    //   self_attn.pos_bias_v                 [8, 128]
    //   self_attn.linear_out.weight          [1024, 1024]
    //   norm_conv.{weight,bias}              LayerNorm
    //   conv.pointwise_conv1.weight          [2048, 1024, 1]
    //   conv.depthwise_conv.{weight,bias}    [1024, 1, 9] + [1024]
    //   conv.pointwise_conv2.weight          [1024, 1024, 1]
    //   norm_feed_forward2.{weight,bias}     LayerNorm
    //   feed_forward2.linear1.weight         [1024, 4096]
    //   feed_forward2.linear2.weight         [4096, 1024]
    //   norm_out.{weight,bias}               LayerNorm
    // ---------------------------------------------------------------------------
    struct ConformerBlock {
        // FF1
        half *ff1_ln_w = nullptr, *ff1_ln_b = nullptr;
        half *ff1_w1 = nullptr;   // [1024, 4096]
        half *ff1_w2 = nullptr;   // [4096, 1024]

        // MHSA
        half *mhsa_ln_w = nullptr, *mhsa_ln_b = nullptr;
        half *q_w = nullptr;      // [1024, 1024]
        half *k_w = nullptr;      // [1024, 1024]
        half *v_w = nullptr;      // [1024, 1024]
        half *pos_w = nullptr;    // [1024, 1024] (linear_pos)
        half *pos_bias_u = nullptr;  // [8, 128]
        half *pos_bias_v = nullptr;  // [8, 128]
        half *out_w = nullptr;    // [1024, 1024]

        // Conv module
        half *conv_ln_w = nullptr, *conv_ln_b = nullptr;
        half *conv_pw1_w = nullptr;   // [2048, 1024, 1]
        half *conv_dw_w = nullptr;    // [1024, 1, 9]
        half *conv_dw_b = nullptr;    // [1024]
        half *conv_pw2_w = nullptr;   // [1024, 1024, 1]

        // FF2
        half *ff2_ln_w = nullptr, *ff2_ln_b = nullptr;
        half *ff2_w1 = nullptr;   // [1024, 4096]
        half *ff2_w2 = nullptr;   // [4096, 1024]

        // Final LN
        half *final_ln_w = nullptr, *final_ln_b = nullptr;
    } blocks[24];

    // ---------------------------------------------------------------------------
    // Decoder (prediction network)
    //   ONNX LSTM format: weight_ih [1, 4*H, input], weight_hh [1, 4*H, H],
    //   bias [1, 8*H] (concatenated input+recurrent biases)
    // ---------------------------------------------------------------------------
    half *embed_w = nullptr;  // [1025, 640]

    // LSTM layer 0 (no suffix in name)
    half *lstm0_w_ih = nullptr;  // [1, 2560, 640]
    half *lstm0_w_hh = nullptr;  // [1, 2560, 640]
    half *lstm0_bias = nullptr;  // [1, 5120]

    // LSTM layer 1 (suffix ".1" in name)
    half *lstm1_w_ih = nullptr;  // [1, 2560, 640]
    half *lstm1_w_hh = nullptr;  // [1, 2560, 640]
    half *lstm1_bias = nullptr;  // [1, 5120]

    // ---------------------------------------------------------------------------
    // Joint network
    // ---------------------------------------------------------------------------
    half *enc_proj_w = nullptr, *enc_proj_b = nullptr;   // [1024, 640] + [640]
    half *dec_proj_w = nullptr, *dec_proj_b = nullptr;   // [640, 640] + [640]
    half *out_proj_w = nullptr, *out_proj_b = nullptr;   // [640, 1030] + [1030]

    // ---------------------------------------------------------------------------
    // Methods
    // ---------------------------------------------------------------------------

    /// Load weights from a .bin file produced by export_weights.py.
    static Weights load(const std::string& path, cudaStream_t stream = nullptr);

    /// Phase 1: mmap + parse header + populate pages (CPU only, no CUDA needed).
    /// Call upload() after CUDA context is ready.
    static Weights prefetch(const std::string& path);

    /// Phase 2: cudaMalloc + cudaMemcpy from prefetched mmap, then assign pointers.
    void upload(cudaStream_t stream = nullptr);

    /// Free the GPU allocation.
    void free();

    /// Get a GPU pointer for a named tensor (nullptr if not found).
    half* get(const std::string& name) const;

    /// Print diagnostic info.
    void print_info() const;
};

// ---------------------------------------------------------------------------
// CudaModel — encoder + decoder forward pass using cuBLAS + custom kernels
// ---------------------------------------------------------------------------

// Model constants
static constexpr int D_MODEL    = 1024;
static constexpr int D_FF       = 4096;
static constexpr int N_HEADS    = 8;
static constexpr int HEAD_DIM   = D_MODEL / N_HEADS;  // 128
static constexpr int N_BLOCKS   = 24;
static constexpr int D_CONV_PW  = 2048;
static constexpr int CONV_K     = 9;
static constexpr int SUB_CHANNELS = 256;
static constexpr int D_PRED     = 640;   // decoder hidden
static constexpr int N_VOCAB    = 1025;  // vocab size (0..1024)
static constexpr int D_JOINT    = 640;
static constexpr int D_OUTPUT   = 1030;  // joint output (vocab + durations)

struct CudaModel {
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublaslt = nullptr;
    void* lt_workspace = nullptr;
    size_t lt_workspace_size = 0;
    cudaStream_t   stream = nullptr;
    const Weights* w = nullptr;

    // Max sequence length (after 8x downsampling)
    int T_max = 0;

    // --- Pre-concatenated QKV weights per block ---
    half* qkv_w[N_BLOCKS];       // [D_MODEL, 3*D_MODEL] per block

    // --- Pre-combined LSTM weights (W_ih||W_hh side by side) ---
    half* lstm_combined_w[2];    // [4*D_PRED, 2*D_PRED] per layer
    half* lstm_combined_bias[2]; // [4*D_PRED] per layer (b_ih + b_hh pre-added)
    half* lstm_input;            // [2*D_PRED] runtime concat buffer

    // --- Pooled GPU allocation (single cudaMalloc for all buffers below) ---
    void* gpu_pool = nullptr;

    // --- Encoder buffers (all FP16, carved from gpu_pool) ---
    float* mel_fp32  = nullptr;  // [128, T_mel] — upload buffer for FP32 mel
    half* mel_fp16   = nullptr;  // [128, T_mel] — cast from FP32 input
    half* sub_buf[2];            // subsampling ping-pong, sized for max intermediate
    half* x          = nullptr;  // [T', D_MODEL] — main encoder activation
    half* ln_out     = nullptr;  // [T', D_MODEL]
    half* ff_mid     = nullptr;  // [T', D_FF]
    half* ff_out     = nullptr;  // [T', D_MODEL]
    half* qkv        = nullptr;  // [T', 3*D_MODEL] — fused QKV output
    half* q          = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half* k          = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half* v          = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half* pos_enc    = nullptr;  // [2*T_max-1, D_MODEL]
    half* pos_proj   = nullptr;  // [2*T_max-1, D_MODEL]
    half* q_u        = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half* q_v_buf    = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half* scores     = nullptr;  // [N_HEADS, T', T']
    half* pos_scores = nullptr;  // [N_HEADS, T', 2*T'-1]
    half* attn_out   = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half* mhsa_out   = nullptr;  // [T', D_MODEL]
    half* conv_mid   = nullptr;  // [T', D_CONV_PW]
    half* conv_glu   = nullptr;  // [T', D_MODEL]
    half* conv_dw    = nullptr;  // [T', D_MODEL]

    // --- Decoder buffers (FP16) ---
    half* lstm_gates    = nullptr;  // [4*D_PRED]
    half* lstm_h[2];                // h state for 2 LSTM layers, each [D_PRED]
    half* lstm_c[2];                // c state for 2 LSTM layers, each [D_PRED]
    half* lstm_h_out[2];            // output h state (uncommitted)
    half* lstm_c_out[2];            // output c state (uncommitted)
    half* enc_proj_all  = nullptr;  // [T_max, D_JOINT] — precomputed after encoding
    half* dec_proj_buf  = nullptr;  // [D_JOINT]
    half* joint_act     = nullptr;  // [D_JOINT]
    half* joint_out     = nullptr;  // [D_OUTPUT]
    int*  argmax_out    = nullptr;  // [2] — token, step (for GPU argmax)

    // --- FP8 quantized weights (carved from fp8_pool) ---
    static constexpr int N_FP8_SCALES = N_BLOCKS * 9 + 6;  // 222
    void* fp8_pool = nullptr;
    uint8_t* fp8_qkv_w[N_BLOCKS];       // [D_MODEL, 3*D_MODEL] per block
    uint8_t* fp8_ff1_w1[N_BLOCKS];      // [D_MODEL, D_FF]
    uint8_t* fp8_ff1_w2[N_BLOCKS];      // [D_FF, D_MODEL]
    uint8_t* fp8_ff2_w1[N_BLOCKS];      // [D_MODEL, D_FF]
    uint8_t* fp8_ff2_w2[N_BLOCKS];      // [D_FF, D_MODEL]
    uint8_t* fp8_pos_w[N_BLOCKS];       // [D_MODEL, D_MODEL]
    uint8_t* fp8_out_w[N_BLOCKS];       // [D_MODEL, D_MODEL]
    uint8_t* fp8_conv_pw1_w[N_BLOCKS];  // [D_CONV_PW, D_MODEL]
    uint8_t* fp8_conv_pw2_w[N_BLOCKS];  // [D_MODEL, D_MODEL]
    uint8_t* fp8_sub_out_w = nullptr;   // [SUB_CHANNELS*W, D_MODEL]
    uint8_t* fp8_enc_proj_w = nullptr;  // [D_MODEL, D_JOINT]
    uint8_t* fp8_lstm_combined_w[2];    // [4*D_PRED, 2*D_PRED]
    uint8_t* fp8_dec_proj_w = nullptr;  // [D_PRED, D_JOINT]
    uint8_t* fp8_out_proj_w = nullptr;  // [D_JOINT, D_OUTPUT]
    float* fp8_scales = nullptr;        // [N_FP8_SCALES] on GPU
    uint8_t* fp8_act_buf = nullptr;     // activation quantization temp
    int* fp8_amax_buf = nullptr;        // [1] scratch for atomicMax in quantize kernel

    // Per-activation-site cached scales for calibrated mode
    // Sites: 9 per block (ff1_w1, ff1_w2, qkv, pos, out, conv_pw1, conv_pw2, ff2_w1, ff2_w2)
    //        + sub_out + enc_proj = N_BLOCKS*9 + 2
    static constexpr int N_FP8_ACT_SITES = N_BLOCKS * 9 + 2;  // 218
    float* fp8_act_site_scales = nullptr;  // [N_FP8_ACT_SITES] on GPU
    bool fp8_calibrated = false;

    // --- Methods ---
    void init(const Weights& weights, cudaStream_t s, int max_mel_frames);
    void free();

    /// Run encoder: mel FP32 [128, T_mel] → encoder output [T', D_MODEL] in this->x
    /// Returns T' (number of encoder frames after 8x downsampling).
    int encode(const float* mel_fp32, int T_mel);

    /// Run encoder from mel_fp32 already on GPU (skip host→device upload).
    /// mel_fp32 member must be pre-populated with [128, T_mel] float data.
    int encode_gpu(int T_mel);

    /// Reset decoder LSTM states to zero (call before greedy decode loop).
    void decoder_reset();

    /// Run one decoder step: given encoder frame index t and previous token,
    /// produce joint output logits. Returns pointer to D_OUTPUT FP16 values on GPU.
    /// Does NOT commit LSTM state — call decoder_commit() if token is non-blank.
    half* decode_step(int enc_frame_idx, int prev_token);

    /// Commit LSTM state after a non-blank token emission.
    void decoder_commit();
};
