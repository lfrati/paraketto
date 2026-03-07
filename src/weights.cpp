// weights.cpp — Weight loading shared by FP16 and FP8 backends
//
// weights.bin format:
//   uint32 magic   = WEIGHTS_MAGIC (0x544B5250 "PRKT")
//   uint32 version = WEIGHTS_VERSION (2)
//   [raw FP16 tensors, 256-byte aligned, in the fixed order below]
//
// The layout defined here IS the file format. No separate index or header.
// Changing this layout requires regenerating weights.bin (run repack_weights.py).

#include "conformer.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Weight layout — assigns pointers and computes total GPU allocation size.
//
// Tensors are placed in a fixed order with 256-byte alignment between them.
// This order matches export_weights.py / repack_weights.py exactly.
// Returns total bytes needed (== file size minus the 8-byte header).
// ---------------------------------------------------------------------------

static size_t assign_weight_pointers(Weights& w) {
    uint8_t* base = (uint8_t*)w.gpu_data;  // null during size computation
    size_t off = 0;

    auto take = [&](half*& ptr, size_t n) {
        off = (off + 255) & ~(size_t)255;
        if (base) ptr = (half*)(base + off);
        off += n * sizeof(half);
    };

    // --- Subsampling (pre_encode) ---
    // conv.0, .2, .5: depthwise [256, 1, 3, 3] = 256*9 elements
    // conv.3, .6:     pointwise [256, 256, 1, 1] = 256*256 elements
    take(w.sub_conv[0].weight, SUB_CHANNELS * 9);
    take(w.sub_conv[0].bias,   SUB_CHANNELS);
    take(w.sub_conv[2].weight, SUB_CHANNELS * 9);
    take(w.sub_conv[2].bias,   SUB_CHANNELS);
    take(w.sub_conv[3].weight, (size_t)SUB_CHANNELS * SUB_CHANNELS);
    take(w.sub_conv[3].bias,   SUB_CHANNELS);
    take(w.sub_conv[5].weight, SUB_CHANNELS * 9);
    take(w.sub_conv[5].bias,   SUB_CHANNELS);
    take(w.sub_conv[6].weight, (size_t)SUB_CHANNELS * SUB_CHANNELS);
    take(w.sub_conv[6].bias,   SUB_CHANNELS);
    take(w.sub_out_w,          (size_t)SUB_CHANNELS * 16 * D_MODEL);  // [4096, 1024]
    take(w.sub_out_b,          D_MODEL);

    // --- 24 conformer blocks ---
    for (int i = 0; i < N_BLOCKS; i++) {
        auto& blk = w.blocks[i];
        take(blk.ff1_ln_w,   D_MODEL);
        take(blk.ff1_ln_b,   D_MODEL);
        take(blk.ff1_w1,     (size_t)D_MODEL * D_FF);    // [1024, 4096]
        take(blk.ff1_w2,     (size_t)D_FF   * D_MODEL);  // [4096, 1024]
        take(blk.mhsa_ln_w,  D_MODEL);
        take(blk.mhsa_ln_b,  D_MODEL);
        take(blk.q_w,        (size_t)D_MODEL * D_MODEL);
        take(blk.k_w,        (size_t)D_MODEL * D_MODEL);
        take(blk.v_w,        (size_t)D_MODEL * D_MODEL);
        take(blk.pos_w,      (size_t)D_MODEL * D_MODEL);
        take(blk.pos_bias_u, (size_t)N_HEADS * HEAD_DIM);  // [8, 128]
        take(blk.pos_bias_v, (size_t)N_HEADS * HEAD_DIM);
        take(blk.out_w,      (size_t)D_MODEL * D_MODEL);
        take(blk.conv_ln_w,  D_MODEL);
        take(blk.conv_ln_b,  D_MODEL);
        take(blk.conv_pw1_w, (size_t)D_CONV_PW * D_MODEL);  // [2048, 1024]
        take(blk.conv_dw_w,  (size_t)D_MODEL   * CONV_K);   // [1024, 9]
        take(blk.conv_dw_b,  D_MODEL);
        take(blk.conv_pw2_w, (size_t)D_MODEL   * D_MODEL);
        take(blk.ff2_ln_w,   D_MODEL);
        take(blk.ff2_ln_b,   D_MODEL);
        take(blk.ff2_w1,     (size_t)D_MODEL * D_FF);
        take(blk.ff2_w2,     (size_t)D_FF   * D_MODEL);
        take(blk.final_ln_w, D_MODEL);
        take(blk.final_ln_b, D_MODEL);
    }

    // --- Decoder ---
    take(w.embed_w,    (size_t)N_VOCAB * D_PRED);        // [1025, 640]
    take(w.lstm0_w_ih, (size_t)4 * D_PRED * D_PRED);    // [1, 2560, 640]
    take(w.lstm0_w_hh, (size_t)4 * D_PRED * D_PRED);
    take(w.lstm0_bias,  8 * D_PRED);                     // [1, 5120] = b_ih||b_hh
    take(w.lstm1_w_ih, (size_t)4 * D_PRED * D_PRED);
    take(w.lstm1_w_hh, (size_t)4 * D_PRED * D_PRED);
    take(w.lstm1_bias,  8 * D_PRED);

    // --- Joint network ---
    take(w.enc_proj_w, (size_t)D_MODEL * D_JOINT);   // [1024, 640]
    take(w.enc_proj_b, D_JOINT);
    take(w.dec_proj_w, (size_t)D_PRED  * D_JOINT);   // [640, 640]
    take(w.dec_proj_b, D_JOINT);
    take(w.out_proj_w, (size_t)D_JOINT * D_OUTPUT);  // [640, 1030]
    take(w.out_proj_b, D_OUTPUT);

    off = (off + 255) & ~(size_t)255;
    return off;
}

// ---------------------------------------------------------------------------
// Weights::prefetch — mmap weights.bin (CPU only, no CUDA)
// ---------------------------------------------------------------------------

Weights Weights::prefetch(const std::string& path, bool populate) {
    Weights w;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open weights: %s\n", path.c_str());
        std::exit(1);
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;
    int flags = MAP_PRIVATE | (populate ? MAP_POPULATE : 0);
    void* mapped = mmap(nullptr, file_size, PROT_READ, flags, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", path.c_str());
        std::exit(1);
    }
    if (populate) madvise(mapped, file_size, MADV_SEQUENTIAL);

    // Validate header
    const uint8_t* base = (const uint8_t*)mapped;
    uint32_t magic, version;
    memcpy(&magic,   base,     4);
    memcpy(&version, base + 4, 4);
    if (magic != WEIGHTS_MAGIC || version != WEIGHTS_VERSION) {
        fprintf(stderr, "Bad weights file %s (magic=0x%08x version=%u, expected magic=0x%08x version=%u)\n",
                path.c_str(), magic, version, WEIGHTS_MAGIC, WEIGHTS_VERSION);
        munmap(mapped, file_size);
        std::exit(1);
    }

    w.mmap_ptr  = mapped;
    w.mmap_size = file_size;
    return w;
}

// ---------------------------------------------------------------------------
// Weights::from_embedded — point at in-memory data (static binary)
// ---------------------------------------------------------------------------

Weights Weights::from_embedded(const uint8_t* data, size_t size) {
    Weights w;
    // Validate header
    uint32_t magic, version;
    memcpy(&magic,   data,     4);
    memcpy(&version, data + 4, 4);
    if (magic != WEIGHTS_MAGIC || version != WEIGHTS_VERSION) {
        fprintf(stderr, "Bad embedded weights (magic=0x%08x version=%u)\n", magic, version);
        std::exit(1);
    }
    w.embedded_ptr = data;
    w.mmap_size    = size;
    return w;
}

// ---------------------------------------------------------------------------
// Weights::upload — cudaMalloc + copy from prefetched/embedded data
// ---------------------------------------------------------------------------

void Weights::upload(cudaStream_t stream) {
    const uint8_t* src = embedded_ptr ? embedded_ptr : (const uint8_t*)mmap_ptr;
    const uint8_t* data = src + WEIGHTS_HEADER;  // skip 8-byte header

    // Compute GPU allocation size from layout
    gpu_data_size = assign_weight_pointers(*this);  // gpu_data=null → compute only
    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_data_size));

    // Validate file data size
    size_t expected_file = WEIGHTS_HEADER + gpu_data_size;
    if (mmap_size != expected_file) {
        fprintf(stderr, "Weight file size mismatch: expected %zu, got %zu\n",
                expected_file, mmap_size);
        std::exit(1);
    }

    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_data, data, gpu_data_size,
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(gpu_data, data, gpu_data_size,
                               cudaMemcpyHostToDevice));
    }

    if (mmap_ptr) {
        munmap(mmap_ptr, mmap_size);
        mmap_ptr  = nullptr;
        mmap_size = 0;
    }
    embedded_ptr = nullptr;

    assign_weight_pointers(*this);  // assign real pointers into gpu_data
}

// ---------------------------------------------------------------------------
// Weights::allocate_only — FP8 path: malloc without data upload
// ---------------------------------------------------------------------------

void Weights::allocate_only() {
    gpu_data_size = assign_weight_pointers(*this);  // compute with gpu_data=null
    CUDA_CHECK(cudaMalloc(&gpu_data, gpu_data_size));

    if (mmap_ptr) {
        munmap(mmap_ptr, mmap_size);
        mmap_ptr  = nullptr;
        mmap_size = 0;
    }
    embedded_ptr = nullptr;

    assign_weight_pointers(*this);  // assign pointers into gpu_data
}

// ---------------------------------------------------------------------------
// Weights::free
// ---------------------------------------------------------------------------

void Weights::free() {
    if (gpu_data) {
        cudaFree(gpu_data);
        gpu_data      = nullptr;
        gpu_data_size = 0;
    }
}
