// conformer_fp8.h — FP8 CudaModel (cublasLt E4M3 backend)
//
// Included via -include flag for the FP8 build, overriding conformer.h.
// Defines PARAKETTO_FP8 and extends CudaModel with FP8 quantization fields.

#ifndef CONFORMER_H_
#define CONFORMER_H_
#define PARAKETTO_FP8 1

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ---------------------------------------------------------------------------
// Weight file format
// ---------------------------------------------------------------------------
//
// paraketto-fp16.bin (FP16 source, needed only for first-run FP8 quantization):
//   uint32 magic   = 0x544B5250 ("PRKT")
//   uint32 version = 2
//   [raw FP16 tensor data, 256-byte aligned, fixed layout from source]
//
// paraketto-fp8.bin (self-contained FP8 weights, all needed data):
//   char[8] magic  = "PRKTFP8\0"
//   uint32  version = 1
//   uint32  pad     = 0
//   [fp8_pool blob: FP8 quantized weights + scales, pool layout, single cudaMemcpy]
//   [non-GEMM FP16 blob: LN, biases, conv_dw, embed, LSTM, decoder — packed]
//
// Tensor layout for both files is defined in source (weights.cpp / conformer_fp8.cpp).

static constexpr uint32_t WEIGHTS_MAGIC    = 0x544B5250;  // "PRKT"
static constexpr uint32_t WEIGHTS_VERSION  = 2;
static constexpr size_t   WEIGHTS_HEADER   = 8;  // magic(4) + version(4)

static constexpr uint32_t FP8_WEIGHTS_VERSION = 1;
static constexpr size_t   FP8_WEIGHTS_HEADER  = 16;  // magic(8) + version(4) + pad(4)

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------

static constexpr int D_MODEL    = 1024;
static constexpr int D_FF       = 4096;
static constexpr int N_HEADS    = 8;
static constexpr int HEAD_DIM   = D_MODEL / N_HEADS;
static constexpr int N_BLOCKS   = 24;
static constexpr int D_CONV_PW  = 2048;
static constexpr int CONV_K     = 9;
static constexpr int SUB_CHANNELS = 256;
static constexpr int D_PRED     = 640;
static constexpr int N_VOCAB    = 1025;
static constexpr int D_JOINT    = 640;
static constexpr int D_OUTPUT   = 1030;

// ---------------------------------------------------------------------------
// Weights struct
// ---------------------------------------------------------------------------

struct Weights {
    void* gpu_data = nullptr;
    size_t gpu_data_size = 0;

    void*          mmap_ptr     = nullptr;
    size_t         mmap_size    = 0;
    const uint8_t* embedded_ptr = nullptr;

    struct SubConv {
        half *weight = nullptr, *bias = nullptr;
    };
    SubConv sub_conv[7];
    half *sub_out_w = nullptr, *sub_out_b = nullptr;

    struct ConformerBlock {
        half *ff1_ln_w = nullptr, *ff1_ln_b = nullptr;
        half *ff1_w1   = nullptr;
        half *ff1_w2   = nullptr;

        half *mhsa_ln_w  = nullptr, *mhsa_ln_b  = nullptr;
        half *q_w        = nullptr;
        half *k_w        = nullptr;
        half *v_w        = nullptr;
        half *pos_w      = nullptr;
        half *pos_bias_u = nullptr;
        half *pos_bias_v = nullptr;
        half *out_w      = nullptr;

        half *conv_ln_w  = nullptr, *conv_ln_b  = nullptr;
        half *conv_pw1_w = nullptr;
        half *conv_dw_w  = nullptr;
        half *conv_dw_b  = nullptr;
        half *conv_pw2_w = nullptr;

        half *ff2_ln_w = nullptr, *ff2_ln_b = nullptr;
        half *ff2_w1   = nullptr;
        half *ff2_w2   = nullptr;

        half *final_ln_w = nullptr, *final_ln_b = nullptr;
    } blocks[24];

    half *embed_w    = nullptr;
    half *lstm0_w_ih = nullptr;
    half *lstm0_w_hh = nullptr;
    half *lstm0_bias = nullptr;
    half *lstm1_w_ih = nullptr;
    half *lstm1_w_hh = nullptr;
    half *lstm1_bias = nullptr;

    half *enc_proj_w = nullptr, *enc_proj_b = nullptr;
    half *dec_proj_w = nullptr, *dec_proj_b = nullptr;
    half *out_proj_w = nullptr, *out_proj_b = nullptr;

    static Weights prefetch(const std::string& path, bool populate = true);
    static Weights from_embedded(const uint8_t* data, size_t size);
    void upload(cudaStream_t stream = nullptr);
    void allocate_only();
    void free();
};

// ---------------------------------------------------------------------------
// CudaModel — FP8 variant with extra quantization fields
// ---------------------------------------------------------------------------

struct CudaModel {
    cudaStream_t   stream = nullptr;
    const Weights* w      = nullptr;

    int T_max = 0;

    half* qkv_w[N_BLOCKS];
    half* lstm_combined_w[2];
    half* lstm_combined_bias[2];
    half* lstm_input;

    void* gpu_pool = nullptr;

    float* mel_fp32    = nullptr;
    half*  mel_fp16    = nullptr;
    half*  sub_buf[2];
    half*  x           = nullptr;
    half*  ln_out      = nullptr;
    half*  ff_mid      = nullptr;
    half*  ff_out      = nullptr;
    half*  qkv         = nullptr;
    half*  q           = nullptr;
    half*  k           = nullptr;
    half*  v           = nullptr;
    half*  pos_enc     = nullptr;
    half*  pos_proj    = nullptr;
    half*  q_u         = nullptr;
    half*  q_v_buf     = nullptr;
    half*  scores      = nullptr;
    half*  pos_scores  = nullptr;
    half*  attn_out    = nullptr;
    half*  mhsa_out    = nullptr;
    half*  conv_mid    = nullptr;
    half*  conv_glu    = nullptr;
    half*  conv_dw     = nullptr;

    half* lstm_gates    = nullptr;
    half* lstm_h[2];
    half* lstm_c[2];
    half* lstm_h_out[2];
    half* lstm_c_out[2];
    half* enc_proj_all  = nullptr;
    half* dec_proj_buf  = nullptr;
    half* joint_act     = nullptr;
    half* joint_out     = nullptr;
    int*  argmax_out    = nullptr;

    // FP8 quantization data
    static constexpr int N_FP8_SCALES    = N_BLOCKS * 9 + 6;   // 222
    static constexpr int N_FP8_ACT_SITES = N_BLOCKS * 9 + 2;   // 218

    void*    fp8_pool                 = nullptr;
    uint8_t* fp8_qkv_w[N_BLOCKS]     = {};
    uint8_t* fp8_ff1_w1[N_BLOCKS]    = {};
    uint8_t* fp8_ff1_w2[N_BLOCKS]    = {};
    uint8_t* fp8_ff2_w1[N_BLOCKS]    = {};
    uint8_t* fp8_ff2_w2[N_BLOCKS]    = {};
    uint8_t* fp8_pos_w[N_BLOCKS]     = {};
    uint8_t* fp8_out_w[N_BLOCKS]     = {};
    uint8_t* fp8_conv_pw1_w[N_BLOCKS] = {};
    uint8_t* fp8_conv_pw2_w[N_BLOCKS] = {};
    uint8_t* fp8_sub_out_w            = nullptr;
    uint8_t* fp8_enc_proj_w           = nullptr;
    uint8_t* fp8_lstm_combined_w[2]   = {};
    uint8_t* fp8_dec_proj_w           = nullptr;
    uint8_t* fp8_out_proj_w           = nullptr;
    float*   fp8_scales               = nullptr;  // [N_FP8_SCALES]
    float*   fp8_act_site_scales      = nullptr;  // [N_FP8_ACT_SITES]
    uint8_t* fp8_act_buf              = nullptr;
    int*     fp8_amax_buf             = nullptr;
    bool     fp8_calibrated           = false;

    cublasHandle_t   cublas            = nullptr;
    cublasLtHandle_t cublaslt          = nullptr;
    void*            lt_workspace      = nullptr;
    size_t           lt_workspace_size = 0;

    /// fp8_path: path to paraketto-fp8.bin (load if exists, save after quantization).
    /// Pass "embedded" when paraketto-fp8.bin is compiled in (EMBEDDED_WEIGHTS build).
    /// fp8_prefetch: pre-populated mmap of paraketto-fp8.bin (from background prefetch thread).
    void init(const Weights& weights, cudaStream_t s, int max_mel_frames,
              const char* fp8_path = nullptr,
              const void* fp8_prefetch = nullptr, size_t fp8_prefetch_size = 0);
    void free();

    int   encode_gpu(int T_mel);
    void  decoder_reset();
    half* decode_step(int enc_frame_idx, int prev_token);
    void  decoder_commit();
};

#endif  // CONFORMER_H_
