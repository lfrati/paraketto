// conformer.h — Weight loading + CUDA inference for Parakeet conformer
//
// Defines:
//   Weights  — pointers into GPU weight allocation (loaded from weights.bin)
//   CudaModel — pre-allocated buffers + GEMM backend for forward passes

#ifndef CONFORMER_H_
#define CONFORMER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Weight file format
// ---------------------------------------------------------------------------
//
// weights.bin:
//   uint32 magic   = 0x544B5250 ("PRKT" little-endian)
//   uint32 version = 2
//   [raw FP16 tensor data, 256-byte aligned, fixed order matching source layout]
//
// Tensor order and sizes are defined by assign_weight_pointers() in weights.cpp.
// The file layout IS the GPU layout — a single cudaMemcpy loads the entire file.

static constexpr uint32_t WEIGHTS_MAGIC   = 0x544B5250;  // "PRKT"
static constexpr uint32_t WEIGHTS_VERSION = 2;
static constexpr size_t   WEIGHTS_HEADER  = 8;  // magic(4) + version(4)

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------

static constexpr int D_MODEL    = 1024;
static constexpr int D_FF       = 4096;
static constexpr int N_HEADS    = 8;
static constexpr int HEAD_DIM   = D_MODEL / N_HEADS;  // 128
static constexpr int N_BLOCKS   = 24;
static constexpr int D_CONV_PW  = 2048;
static constexpr int CONV_K     = 9;
static constexpr int SUB_CHANNELS = 256;
static constexpr int D_PRED     = 640;   // decoder hidden dim
static constexpr int N_VOCAB    = 1025;  // vocab size (0..1024)
static constexpr int D_JOINT    = 640;
static constexpr int D_OUTPUT   = 1030;  // joint output (vocab + durations)

// ---------------------------------------------------------------------------
// Weights struct — all model weight pointers into a single GPU allocation
// ---------------------------------------------------------------------------
//
// Subsampling (pre_encode):
//   conv.0, .2, .5: depthwise  [SUB_CHANNELS, 1, 3, 3] = 256*9 elements
//   conv.3, .6:     pointwise  [SUB_CHANNELS, SUB_CHANNELS, 1, 1] = 256*256 elements
//   out:            linear     [D_MODEL, SUB_CHANNELS*16] = 1024*4096 elements
//
// Conformer block (x24):
//   ff1_ln, mhsa_ln, conv_ln, ff2_ln, final_ln:  LayerNorm [D_MODEL] weight+bias
//   ff1_w1 [D_MODEL, D_FF], ff1_w2 [D_FF, D_MODEL]
//   q_w, k_w, v_w, pos_w, out_w:  [D_MODEL, D_MODEL]
//   pos_bias_u, pos_bias_v:  [N_HEADS, HEAD_DIM]
//   conv_pw1_w [D_CONV_PW, D_MODEL], conv_dw_w [D_MODEL, CONV_K], conv_pw2_w [D_MODEL, D_MODEL]
//   ff2_w1 [D_MODEL, D_FF], ff2_w2 [D_FF, D_MODEL]
//
// Decoder:
//   embed_w [N_VOCAB, D_PRED]
//   lstm{0,1}_w_ih [1, 4*D_PRED, D_PRED], lstm{0,1}_w_hh [1, 4*D_PRED, D_PRED]
//   lstm{0,1}_bias [1, 8*D_PRED]  (b_ih + b_hh concatenated)
//
// Joint network:
//   enc_proj_w [D_MODEL, D_JOINT], enc_proj_b [D_JOINT]
//   dec_proj_w [D_PRED, D_JOINT],  dec_proj_b [D_JOINT]
//   out_proj_w [D_JOINT, D_OUTPUT], out_proj_b [D_OUTPUT]

struct Weights {
    void* gpu_data = nullptr;
    size_t gpu_data_size = 0;

    // Prefetch state (temporary — cleared after upload)
    void*          mmap_ptr     = nullptr;
    size_t         mmap_size    = 0;
    const uint8_t* embedded_ptr = nullptr;

    // ---------------------------------------------------------------------------
    // Subsampling (pre_encode)
    // ---------------------------------------------------------------------------
    struct SubConv {
        half *weight = nullptr, *bias = nullptr;
    };
    SubConv sub_conv[7];  // indices 0, 2, 3, 5, 6 used; 1 and 4 unused
    half *sub_out_w = nullptr, *sub_out_b = nullptr;

    // ---------------------------------------------------------------------------
    // Conformer blocks (x24)
    // ---------------------------------------------------------------------------
    struct ConformerBlock {
        half *ff1_ln_w = nullptr, *ff1_ln_b = nullptr;
        half *ff1_w1   = nullptr;   // [D_MODEL, D_FF]
        half *ff1_w2   = nullptr;   // [D_FF, D_MODEL]

        half *mhsa_ln_w  = nullptr, *mhsa_ln_b  = nullptr;
        half *q_w        = nullptr;  // [D_MODEL, D_MODEL]
        half *k_w        = nullptr;
        half *v_w        = nullptr;
        half *pos_w      = nullptr;
        half *pos_bias_u = nullptr;  // [N_HEADS, HEAD_DIM]
        half *pos_bias_v = nullptr;
        half *out_w      = nullptr;

        half *conv_ln_w  = nullptr, *conv_ln_b  = nullptr;
        half *conv_pw1_w = nullptr;  // [D_CONV_PW, D_MODEL]
        half *conv_dw_w  = nullptr;  // [D_MODEL, CONV_K]
        half *conv_dw_b  = nullptr;  // [D_MODEL]
        half *conv_pw2_w = nullptr;  // [D_MODEL, D_MODEL]

        half *ff2_ln_w   = nullptr, *ff2_ln_b   = nullptr;
        half *ff2_w1     = nullptr;
        half *ff2_w2     = nullptr;

        half *final_ln_w = nullptr, *final_ln_b = nullptr;
    } blocks[24];

    // ---------------------------------------------------------------------------
    // Decoder
    // ---------------------------------------------------------------------------
    half *embed_w    = nullptr;  // [N_VOCAB, D_PRED]

    half *lstm0_w_ih = nullptr;  // [1, 4*D_PRED, D_PRED]
    half *lstm0_w_hh = nullptr;
    half *lstm0_bias = nullptr;  // [1, 8*D_PRED]

    half *lstm1_w_ih = nullptr;
    half *lstm1_w_hh = nullptr;
    half *lstm1_bias = nullptr;

    // ---------------------------------------------------------------------------
    // Joint network
    // ---------------------------------------------------------------------------
    half *enc_proj_w = nullptr, *enc_proj_b = nullptr;
    half *dec_proj_w = nullptr, *dec_proj_b = nullptr;
    half *out_proj_w = nullptr, *out_proj_b = nullptr;

    // ---------------------------------------------------------------------------
    // Methods
    // ---------------------------------------------------------------------------

    /// Prefetch: mmap weights.bin (CPU only, no CUDA).
    /// populate=false skips MAP_POPULATE (FP8 path: only GPU layout needed).
    static Weights prefetch(const std::string& path, bool populate = true);

    /// Embedded variant: point at in-memory data (no mmap).
    static Weights from_embedded(const uint8_t* data, size_t size);

    /// Upload prefetched/embedded data to GPU, assign weight pointers.
    void upload(cudaStream_t stream = nullptr);

    /// FP8 path: cudaMalloc + assign pointers only (no data upload).
    /// fp8_load() will populate GPU memory from weights_fp8.bin.
    void allocate_only();

    /// Free the GPU allocation.
    void free();
};

// ---------------------------------------------------------------------------
// CudaModel — encoder + decoder forward pass
// ---------------------------------------------------------------------------

struct CudaModel {
    cudaStream_t   stream = nullptr;
    const Weights* w      = nullptr;

    int T_max = 0;  // max encoder frames (after 8x downsampling)

    // Pre-concatenated QKV weights per block [D_MODEL, 3*D_MODEL]
    half* qkv_w[N_BLOCKS];

    // Pre-combined LSTM weights [4*D_PRED, 2*D_PRED] and biases [4*D_PRED]
    half* lstm_combined_w[2];
    half* lstm_combined_bias[2];
    half* lstm_input;  // [2*D_PRED] runtime concat buffer

    // Single pooled GPU allocation for all inference buffers
    void* gpu_pool = nullptr;

    // Encoder buffers
    float* mel_fp32    = nullptr;  // [128, T_mel]
    half*  mel_fp16    = nullptr;  // [128, T_mel]
    half*  sub_buf[2];             // subsampling ping-pong
    half*  x           = nullptr;  // [T', D_MODEL] main activation
    half*  ln_out      = nullptr;
    half*  ff_mid      = nullptr;  // [T', D_FF]
    half*  ff_out      = nullptr;
    half*  qkv         = nullptr;  // [T', 3*D_MODEL]
    half*  q           = nullptr;  // [N_HEADS, T', HEAD_DIM]
    half*  k           = nullptr;
    half*  v           = nullptr;
    half*  pos_enc     = nullptr;  // [2*T_max-1, D_MODEL]
    half*  pos_proj    = nullptr;
    half*  q_u         = nullptr;
    half*  q_v_buf     = nullptr;
    half*  scores      = nullptr;  // [N_HEADS, T', T']
    half*  pos_scores  = nullptr;  // [N_HEADS, T', 2*T'-1]
    half*  attn_out    = nullptr;
    half*  mhsa_out    = nullptr;
    half*  conv_mid    = nullptr;  // [T', D_CONV_PW]
    half*  conv_glu    = nullptr;
    half*  conv_dw     = nullptr;

    // Decoder buffers
    half* lstm_gates    = nullptr;  // [4*D_PRED]
    half* lstm_h[2];                // [D_PRED] per layer
    half* lstm_c[2];
    half* lstm_h_out[2];
    half* lstm_c_out[2];
    half* enc_proj_all  = nullptr;  // [T_max, D_JOINT]
    half* dec_proj_buf  = nullptr;  // [D_JOINT]
    half* joint_act     = nullptr;
    half* joint_out     = nullptr;  // [D_OUTPUT]
    int*  argmax_out    = nullptr;  // [2]: token, step

    void init(const Weights& weights, cudaStream_t s, int max_mel_frames);
    void free();

    int   encode_gpu(int T_mel);
    void  decoder_reset();
    half* decode_step(int enc_frame_idx, int prev_token);
    void  decoder_commit();
};

#endif  // CONFORMER_H_
