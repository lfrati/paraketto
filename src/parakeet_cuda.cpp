// parakeet_cuda.cpp — Custom CUDA/cuBLAS runtime for Parakeet TDT 0.6B V2
//
// Standalone CUDA backend: reads a WAV file, computes mel spectrogram,
// runs encoder (cuBLAS + custom kernels) and TDT greedy decoder on GPU.
// No TensorRT dependency.
//
// Build: make parakeet_cuda
// Usage: ./parakeet_cuda [--weights FILE] audio.wav

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>

#include "conformer.h"
#include "cpp-httplib/httplib.h"

// ---------------------------------------------------------------------------
// Constants — match NeMo Parakeet TDT 0.6B V2 preprocessor
// ---------------------------------------------------------------------------

static constexpr int N_FFT = 512;
static constexpr int HOP = 160;
static constexpr int N_MELS = 128;
static constexpr int N_FREQ = N_FFT / 2 + 1;  // 257
static constexpr float PREEMPH = 0.97f;
static constexpr float LOG_EPS = 5.9604645e-08f;
static constexpr float NORM_EPS = 1e-05f;

// ---------------------------------------------------------------------------
// CUDA helpers
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

struct GpuBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    GpuBuf() = default;
    explicit GpuBuf(size_t n) : bytes(n) { CUDA_CHECK(cudaMalloc(&ptr, n)); }
    ~GpuBuf() { if (ptr) cudaFree(ptr); }
    GpuBuf(const GpuBuf&) = delete;
    GpuBuf& operator=(const GpuBuf&) = delete;
    GpuBuf(GpuBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; }
    GpuBuf& operator=(GpuBuf&& o) noexcept {
        if (ptr) cudaFree(ptr);
        ptr = o.ptr; bytes = o.bytes; o.ptr = nullptr;
        return *this;
    }
    void resize(size_t n) {
        if (n > bytes) {
            if (ptr) cudaFree(ptr);
            bytes = n;
            CUDA_CHECK(cudaMalloc(&ptr, n));
        }
    }
};

// ---------------------------------------------------------------------------
// WAV reader (16kHz mono, int16 or float32)
// ---------------------------------------------------------------------------

struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
};

static WavData read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open WAV: %s\n", path.c_str()); std::exit(1); }

    char riff[4]; f.read(riff, 4);
    if (memcmp(riff, "RIFF", 4)) { fprintf(stderr, "Not RIFF: %s\n", path.c_str()); std::exit(1); }
    uint32_t file_size; f.read((char*)&file_size, 4);
    char wave[4]; f.read(wave, 4);
    if (memcmp(wave, "WAVE", 4)) { fprintf(stderr, "Not WAVE: %s\n", path.c_str()); std::exit(1); }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    std::vector<char> raw_data;

    while (f) {
        char id[4]; uint32_t sz;
        if (!f.read(id, 4) || !f.read((char*)&sz, 4)) break;
        if (!memcmp(id, "fmt ", 4)) {
            f.read((char*)&audio_format, 2); f.read((char*)&num_channels, 2);
            f.read((char*)&sample_rate, 4);
            f.seekg(6, std::ios::cur);  // skip byte_rate + block_align
            f.read((char*)&bits_per_sample, 2);
            if (sz > 16) f.seekg(sz - 16, std::ios::cur);
        } else if (!memcmp(id, "data", 4)) {
            data_size = sz; raw_data.resize(sz); f.read(raw_data.data(), sz);
        } else {
            f.seekg(sz, std::ios::cur);
        }
    }

    if (num_channels != 1) { fprintf(stderr, "Need mono, got %d ch\n", num_channels); std::exit(1); }
    if (sample_rate != 16000) { fprintf(stderr, "Need 16kHz, got %dHz: %s\n(resample with: ffmpeg -i input.wav -ar 16000 output.wav)\n", sample_rate, path.c_str()); std::exit(1); }

    WavData wav;
    wav.sample_rate = sample_rate;
    if (audio_format == 1 && bits_per_sample == 16) {
        int n = data_size / 2; wav.samples.resize(n);
        auto* src = (const int16_t*)raw_data.data();
        for (int i = 0; i < n; i++) wav.samples[i] = src[i] / 32768.0f;
    } else if (audio_format == 3 && bits_per_sample == 32) {
        int n = data_size / 4; wav.samples.resize(n);
        memcpy(wav.samples.data(), raw_data.data(), data_size);
    } else {
        fprintf(stderr, "Unsupported fmt=%d bits=%d\n", audio_format, bits_per_sample); std::exit(1);
    }
    return wav;
}

static WavData read_wav_from_memory(const char* buf, size_t len) {
    if (len < 44) return {};
    if (memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4)) return {};

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    const char* raw_data = nullptr;

    size_t pos = 12;
    while (pos + 8 <= len) {
        const char* id = buf + pos;
        uint32_t sz; memcpy(&sz, buf + pos + 4, 4);
        pos += 8;
        if (!memcmp(id, "fmt ", 4) && pos + 16 <= len) {
            memcpy(&audio_format, buf + pos, 2);
            memcpy(&num_channels, buf + pos + 2, 2);
            memcpy(&sample_rate, buf + pos + 4, 4);
            memcpy(&bits_per_sample, buf + pos + 14, 2);
        } else if (!memcmp(id, "data", 4)) {
            data_size = sz; raw_data = buf + pos;
            if (pos + sz > len) data_size = len - pos;
        }
        pos += sz;
    }

    if (!raw_data || num_channels != 1 || sample_rate != 16000) return {};

    WavData wav;
    wav.sample_rate = sample_rate;
    if (audio_format == 1 && bits_per_sample == 16) {
        int n = data_size / 2; wav.samples.resize(n);
        auto* src = (const int16_t*)raw_data;
        for (int i = 0; i < n; i++) wav.samples[i] = src[i] / 32768.0f;
    } else if (audio_format == 3 && bits_per_sample == 32) {
        int n = data_size / 4; wav.samples.resize(n);
        memcpy(wav.samples.data(), raw_data, data_size);
    } else {
        return {};
    }
    return wav;
}

// ---------------------------------------------------------------------------
// Mel spectrogram (CPU) — matches NeMo nemo128.onnx preprocessor
// ---------------------------------------------------------------------------

static constexpr int N_MEL_ENTRIES = 504;
static const struct { uint16_t freq, mel; float weight; } MEL_FILTERBANK[] = {
    {1,0,2.83775423e-02f},{1,1,1.43890111e-02f},{2,1,1.39885303e-02f},{2,2,2.87780222e-02f},
    {3,3,4.23660800e-02f},{3,4,4.00479796e-04f},{4,4,2.79770531e-02f},{4,5,1.47894993e-02f},
    {5,5,1.35880429e-02f},{5,6,2.91785114e-02f},{6,7,4.19655927e-02f},{6,8,8.00959882e-04f},
    {7,8,2.75765732e-02f},{7,9,1.51899932e-02f},{8,9,1.31875388e-02f},{8,10,2.95790080e-02f},
    {9,11,4.15650979e-02f},{9,12,1.20143977e-03f},{10,12,2.71760952e-02f},
    {10,13,1.55904740e-02f},{11,13,1.27870589e-02f},{11,14,2.99794879e-02f},
    {12,15,4.11646180e-02f},{12,16,1.60191976e-03f},{13,16,2.67756134e-02f},
    {13,17,1.59909539e-02f},{14,17,1.23865791e-02f},{14,18,3.03799864e-02f},
    {15,19,4.07640859e-02f},{15,20,2.00245297e-03f},{16,20,2.63751131e-02f},
    {16,21,1.63914450e-02f},{17,21,1.19861150e-02f},{17,22,3.07804029e-02f},
    {18,23,4.03636880e-02f},{18,24,2.40287953e-03f},{19,24,2.59746537e-02f},
    {19,25,1.67919137e-02f},{20,25,1.15856193e-02f},{20,26,3.11809480e-02f},
    {21,27,3.99631523e-02f},{21,28,2.80341529e-03f},{22,28,2.55741179e-02f},
    {22,29,1.71924494e-02f},{23,29,1.11850835e-02f},{23,30,3.15814428e-02f},
    {24,31,3.95626761e-02f},{24,32,3.20383953e-03f},{25,32,2.51736939e-02f},
    {25,33,1.75928734e-02f},{26,33,1.07846595e-02f},{26,34,3.19819078e-02f},
    {27,35,3.91621925e-02f},{27,36,3.60437529e-03f},{28,36,2.47731581e-02f},
    {28,37,1.79934092e-02f},{29,37,1.03841228e-02f},{29,38,3.23824435e-02f},
    {30,39,3.87616158e-02f},{30,40,4.00489522e-03f},{31,40,2.43727025e-02f},
    {31,41,1.83207151e-02f},{32,41,1.02025867e-02f},{32,42,3.16115171e-02f},
    {33,43,3.85538451e-02f},{33,44,1.65611738e-03f},{34,44,2.83605382e-02f},
    {34,45,1.06703397e-02f},{35,45,1.99555922e-02f},{35,46,1.79612618e-02f},
    {36,46,1.31711997e-02f},{36,47,2.36918963e-02f},{37,47,7.85238389e-03f},
    {37,48,2.80131958e-02f},{38,48,3.85507802e-03f},{38,49,3.10652889e-02f},
    {39,49,1.04670227e-03f},{39,50,3.29767838e-02f},{40,51,3.25256810e-02f},
    {40,52,6.47027802e-04f},{41,52,3.09822969e-02f},{41,53,1.38180249e-03f},
    {42,53,3.02656405e-02f},{42,54,1.32793270e-03f},{43,54,3.02831493e-02f},
    {43,55,5.75242273e-04f},{44,54,8.53460981e-04f},{44,55,2.93036904e-02f},
    {45,55,2.90606264e-03f},{45,56,2.65820120e-02f},{46,56,5.46437269e-03f},
    {46,57,2.33837739e-02f},{47,57,8.45953170e-03f},{47,58,1.97757483e-02f},
    {48,58,1.18284719e-02f},{48,59,1.58191081e-02f},{49,59,1.55134378e-02f},
    {49,60,1.15699172e-02f},{50,60,1.94616914e-02f},{50,61,7.07929302e-03f},
    {51,61,2.36250367e-02f},{51,62,2.39386875e-03f},{52,61,2.62733712e-03f},
    {52,62,2.28912402e-02f},{53,62,7.95065891e-03f},{53,63,1.70882139e-02f},
    {54,63,1.33589162e-02f},{54,64,1.12167532e-02f},{55,64,1.88182276e-02f},
    {55,65,5.30957337e-03f},{56,64,6.49119786e-04f},{56,65,2.30457932e-02f},
    {57,65,6.98436378e-03f},{57,66,1.62973497e-02f},{58,66,1.32723171e-02f},
    {58,67,9.60848108e-03f},{59,67,1.94902495e-02f},{59,68,3.00106104e-03f},
    {60,67,3.76859191e-03f},{60,68,1.83480773e-02f},{61,68,1.06366761e-02f},
    {61,69,1.11188320e-02f},{62,69,1.73619855e-02f},{62,70,4.04149853e-03f},
    {63,69,3.08591709e-03f},{63,70,1.79775134e-02f},{64,70,1.03265326e-02f},
    {64,71,1.04092704e-02f},{65,71,1.73665807e-02f},{65,72,3.04870680e-03f},
    {66,71,4.40334715e-03f},{66,72,1.57030448e-02f},{67,72,1.18445558e-02f},
    {67,73,7.96261150e-03f},{68,73,1.90405622e-02f},{68,74,4.72926622e-04f},
    {69,73,7.26913335e-03f},{69,74,1.19639309e-02f},{70,74,1.47695513e-02f},
    {70,75,4.18809662e-03f},{71,74,3.55228153e-03f},{71,75,1.51380403e-02f},
    {72,75,1.13017363e-02f},{72,76,7.12988246e-03f},{73,75,6.12856937e-04f},
    {73,76,1.75640546e-02f},{74,76,8.56065098e-03f},{74,77,9.37294681e-03f},
    {75,77,1.61811691e-02f},{75,78,1.51162315e-03f},{76,77,6.47500018e-03f},
    {76,78,1.09865637e-02f},{77,78,1.42282611e-02f},{77,79,3.00567481e-03f},
    {78,78,4.97904234e-03f},{78,79,1.20344795e-02f},{79,79,1.28258187e-02f},
    {79,80,3.97208985e-03f},{80,79,4.01224289e-03f},{80,80,1.25756515e-02f},
    {81,80,1.19170034e-02f},{81,81,4.46630735e-03f},{82,80,3.51837068e-03f},
    {82,81,1.26648927e-02f},{83,81,1.14489142e-02f},{83,82,4.53974446e-03f},
    {84,81,3.44560994e-03f},{84,82,1.23523679e-02f},{85,82,1.13728084e-02f},
    {85,83,4.23958898e-03f},{86,82,3.74636892e-03f},{86,83,1.16843004e-02f},
    {87,83,1.16440291e-02f},{87,84,3.60925705e-03f},{88,83,4.37669503e-03f},
    {88,84,1.07034026e-02f},{89,84,1.22213028e-02f},{89,85,2.68887472e-03f},
    {90,84,5.29625546e-03f},{90,85,9.44895484e-03f},{91,85,1.30669065e-02f},
    {91,86,1.51516416e-03f},{92,85,6.46777125e-03f},{92,86,7.95710366e-03f},
    {93,86,1.41456285e-02f},{93,87,1.22195692e-04f},{94,86,7.85722025e-03f},
    {94,87,6.26075501e-03f},{95,86,1.56881078e-03f},{95,87,1.23993140e-02f},
    {96,87,9.43337940e-03f},{96,88,4.39020712e-03f},{97,87,3.44107836e-03f},
    {97,88,1.02397334e-02f},{98,88,1.11677265e-02f},{98,89,2.37318384e-03f},
    {99,88,5.45757031e-03f},{99,89,7.94730987e-03f},{100,89,1.30341724e-02f},
    {100,90,2.34960549e-04f},{101,89,7.59281684e-03f},{101,90,5.54667925e-03f},
    {102,89,2.15146085e-03f},{102,90,1.08583970e-02f},{103,90,9.82361566e-03f},
    {103,91,3.06027057e-03f},{104,90,4.63847397e-03f},{104,91,8.12183693e-03f},
    {105,91,1.21288449e-02f},{105,92,5.08507947e-04f},{106,91,7.18792249e-03f},
    {106,92,5.33170253e-03f},{107,91,2.24700011e-03f},{107,92,1.01548973e-02f},
    {108,92,9.78126191e-03f},{108,93,2.50594132e-03f},{109,92,5.07294899e-03f},
    {109,93,7.10208854e-03f},{110,92,3.64636711e-04f},{110,93,1.16982358e-02f},
    {111,93,7.91501906e-03f},{111,94,4.04057093e-03f},{112,93,3.42838466e-03f},
    {112,94,8.42033420e-03f},{113,94,1.07586123e-02f},{113,95,9.84416809e-04f},
    {114,94,6.48314506e-03f},{114,95,5.15802298e-03f},{115,94,2.20767758e-03f},
    {115,95,9.33162961e-03f},{116,95,9.51629691e-03f},{116,96,1.92344654e-03f},
    {117,95,5.44217229e-03f},{117,96,5.90046262e-03f},{118,95,1.36804674e-03f},
    {118,96,9.87747870e-03f},{119,96,8.63430277e-03f},{119,97,2.51718867e-03f},
    {120,96,4.75208182e-03f},{120,97,6.30693184e-03f},{121,96,8.69860232e-04f},
    {121,97,1.00966739e-02f},{122,97,8.07521679e-03f},{122,98,2.80221645e-03f},
    {123,97,4.37569525e-03f},{123,98,6.41361158e-03f},{124,97,6.76174182e-04f},
    {124,98,1.00250067e-02f},{125,98,7.80408969e-03f},{125,99,2.81232572e-03f},
    {126,98,4.27877204e-03f},{126,99,6.25363085e-03f},{127,98,7.53454107e-04f},
    {127,99,9.69493575e-03f},{128,99,7.78910099e-03f},{128,100,2.57840217e-03f},
    {129,99,4.42979531e-03f},{129,100,5.85767115e-03f},{130,99,1.07048987e-03f},
    {130,100,9.13694035e-03f},{131,100,8.00086837e-03f},{131,101,2.12908420e-03f},
    {132,100,4.79972083e-03f},{132,101,5.25395107e-03f},{133,100,1.59857306e-03f},
    {133,101,8.37881863e-03f},{134,101,8.41220003e-03f},{134,102,1.49072113e-03f},
    {135,101,5.36181452e-03f},{135,102,4.46844148e-03f},{136,101,2.31142947e-03f},
    {136,102,7.44616147e-03f},{137,102,8.99835024e-03f},{137,103,6.87403546e-04f},
    {138,102,6.09152485e-03f},{138,103,3.52498284e-03f},{139,102,3.18469969e-03f},
    {139,103,6.36256207e-03f},{140,102,2.77874351e-04f},{140,103,9.20014083e-03f},
    {141,103,6.96623977e-03f},{141,104,2.44545308e-03f},{142,103,4.19629412e-03f},
    {142,104,5.14938589e-03f},{143,103,1.42634881e-03f},{143,104,7.85331801e-03f},
    {144,104,7.96535145e-03f},{144,105,1.24982581e-03f},{145,104,5.32585150e-03f},
    {145,105,3.82645428e-03f},{146,104,2.68635130e-03f},{146,105,6.40308252e-03f},
    {147,104,4.68511244e-05f},{147,105,8.97971075e-03f},{148,105,6.55481406e-03f},
    {148,106,2.41177063e-03f},{149,105,4.03953623e-03f},{149,106,4.86712391e-03f},
    {150,105,1.52425852e-03f},{150,106,7.32247718e-03f},{151,106,7.86607340e-03f},
    {151,107,9.21842700e-04f},{152,106,5.46925049e-03f},{152,107,3.26154474e-03f},
    {153,106,3.07242735e-03f},{153,107,5.60124684e-03f},{154,106,6.75604504e-04f},
    {154,107,7.94094894e-03f},{155,107,6.96028676e-03f},{155,108,1.60108518e-03f},
    {156,107,4.67632990e-03f},{156,108,3.83062055e-03f},{157,107,2.39237328e-03f},
    {157,108,6.06015604e-03f},{158,107,1.08416571e-04f},{158,108,8.28969106e-03f},
    {159,108,6.32242858e-03f},{159,109,2.02370179e-03f},{160,108,4.14602179e-03f},
    {160,109,4.14825371e-03f},{161,108,1.96961453e-03f},{161,109,6.27280539e-03f},
    {162,109,7.99843483e-03f},{162,110,1.92363179e-04f},{163,109,5.92449680e-03f},
    {163,110,2.21690582e-03f},{164,109,3.85055807e-03f},{164,110,4.24144836e-03f},
    {165,109,1.77661981e-03f},{165,110,6.26599137e-03f},{166,110,7.71697052e-03f},
    {166,111,2.76575185e-04f},{167,110,5.74063556e-03f},{167,111,2.20581912e-03f},
    {168,110,3.76430037e-03f},{168,111,4.13506292e-03f},{169,110,1.78796554e-03f},
    {169,111,6.06430648e-03f},{170,111,7.63017312e-03f},{170,112,1.75219175e-04f},
    {171,111,5.74692758e-03f},{171,112,2.01358437e-03f},{172,111,3.86368297e-03f},
    {172,112,3.85194970e-03f},{173,111,1.98043790e-03f},{173,112,5.69031481e-03f},
    {174,111,9.71930931e-05f},{174,112,7.52868038e-03f},{175,112,5.92159759e-03f},
    {175,113,1.66142290e-03f},{176,112,4.12702141e-03f},{176,113,3.41325696e-03f},
    {177,112,2.33244477e-03f},{177,113,5.16509078e-03f},{178,112,5.37868182e-04f},
    {178,113,6.91692485e-03f},{179,113,6.24442240e-03f},{179,114,1.16902194e-03f},
    {180,113,4.53430973e-03f},{180,114,2.83838250e-03f},{181,113,2.82419752e-03f},
    {181,114,4.50774282e-03f},{182,113,1.11408497e-03f},{182,114,6.17710315e-03f},
    {183,114,6.69668848e-03f},{183,115,5.54416154e-04f},{184,114,5.06713148e-03f},
    {184,115,2.14513764e-03f},{185,114,3.43757519e-03f},{185,115,3.73585895e-03f},
    {186,114,1.80801854e-03f},{186,115,5.32658072e-03f},{187,114,1.78461909e-04f},
    {187,115,6.91730250e-03f},{188,115,5.70873963e-03f},{188,116,1.34983147e-03f},
    {189,115,4.15590871e-03f},{189,116,2.86567095e-03f},{190,115,2.60307803e-03f},
    {190,116,4.38151043e-03f},{191,115,1.05024735e-03f},{191,116,5.89735014e-03f},
    {192,116,6.44365605e-03f},{192,117,4.67512553e-04f},{193,116,4.96392883e-03f},
    {193,117,1.91198511e-03f},{194,116,3.48420208e-03f},{194,117,3.35645746e-03f},
    {195,116,2.00447510e-03f},{195,117,4.80093015e-03f},{196,116,5.24748175e-04f},
    {196,117,6.24540262e-03f},{197,117,5.84763382e-03f},{197,118,8.88335751e-04f},
    {198,117,4.43757791e-03f},{198,118,2.26480025e-03f},{199,117,3.02752224e-03f},
    {199,118,3.64126475e-03f},{200,117,1.61746633e-03f},{200,118,5.01772901e-03f},
    {201,117,2.07410412e-04f},{201,118,6.39419351e-03f},{202,118,5.45063149e-03f},
    {202,119,1.11872482e-03f},{203,118,4.10695281e-03f},{203,119,2.43038684e-03f},
    {204,118,2.76327459e-03f},{204,119,3.74204875e-03f},{205,118,1.41959626e-03f},
    {205,119,5.05371112e-03f},{206,118,7.59178292e-05f},{206,119,6.36537280e-03f},
    {207,119,5.23142610e-03f},{207,120,1.17926707e-03f},{208,119,3.95102799e-03f},
    {208,120,2.42915284e-03f},{209,119,2.67063035e-03f},{209,120,3.67903826e-03f},
    {210,119,1.39023212e-03f},{210,120,4.92892368e-03f},{211,119,1.09834153e-04f},
    {211,120,6.17880980e-03f},{212,120,5.17058233e-03f},{212,121,1.08887581e-03f},
    {213,120,3.95047618e-03f},{213,121,2.27992097e-03f},{214,120,2.73036934e-03f},
    {214,121,3.47096613e-03f},{215,120,1.51026295e-03f},{215,121,4.66201128e-03f},
    {216,120,2.90156546e-04f},{216,121,5.85305644e-03f},{217,121,5.25011867e-03f},
    {217,122,8.65069218e-04f},{218,121,4.08743415e-03f},{218,122,2.00005155e-03f},
    {219,121,2.92475009e-03f},{219,122,3.13503365e-03f},{220,121,1.76206592e-03f},
    {220,122,4.27001575e-03f},{221,121,5.99381863e-04f},{221,122,5.40499808e-03f},
    {222,122,5.45332674e-03f},{222,123,5.23980649e-04f},{223,122,4.34540212e-03f},
    {223,123,1.60550303e-03f},{224,122,3.23747750e-03f},{224,123,2.68702558e-03f},
    {225,122,2.12955265e-03f},{225,123,3.76854767e-03f},{226,122,1.02162780e-03f},
    {226,123,4.85007046e-03f},{227,123,5.76511817e-03f},{227,124,8.02745271e-05f},
    {228,123,4.70935972e-03f},{228,124,1.11088040e-03f},{229,123,3.65360104e-03f},
    {229,124,2.14148615e-03f},{230,123,2.59784260e-03f},{230,124,3.17209191e-03f},
    {231,123,1.54208392e-03f},{231,124,4.20269789e-03f},{232,123,4.86325298e-04f},
    {232,124,5.23330364e-03f},{233,124,5.16542047e-03f},{233,125,5.29693381e-04f},
    {234,124,4.15937044e-03f},{234,125,1.51177205e-03f},{235,124,3.15332110e-03f},
    {235,125,2.49385089e-03f},{236,124,2.14727153e-03f},{236,125,3.47592961e-03f},
    {237,124,1.14122173e-03f},{237,125,4.45800833e-03f},{238,124,1.35172202e-04f},
    {238,125,5.44008752e-03f},{239,125,4.74216696e-03f},{239,126,8.10103375e-04f},
    {240,125,3.78348771e-03f},{240,126,1.74594612e-03f},{241,125,2.82480847e-03f},
    {241,126,2.68178876e-03f},{242,125,1.86612923e-03f},{242,126,3.61763150e-03f},
    {243,125,9.07449867e-04f},{243,126,4.55347402e-03f},{244,126,5.39048947e-03f},
    {244,127,4.76550158e-05f},{245,126,4.47693421e-03f},{245,127,9.39444755e-04f},
    {246,126,3.56337917e-03f},{246,127,1.83123455e-03f},{247,126,2.64982390e-03f},
    {247,127,2.72302423e-03f},{248,126,1.73626863e-03f},{248,127,3.61481425e-03f},
    {249,126,8.22713540e-04f},{249,127,4.50660381e-03f},{250,127,5.22315269e-03f},
    {251,127,4.35261801e-03f},{252,127,3.48208379e-03f},{253,127,2.61154911e-03f},
    {254,127,1.74101465e-03f},{255,127,8.70480086e-04f}
};

// Hann window (400 samples, zero-padded to 512), extracted from nemo128.onnx.
static const float HANN_WINDOW[N_FFT] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,6.19933344e-05f,2.47957942e-04f,5.57847728e-04f,9.91585897e-04f,
    1.54906476e-03f,2.23014620e-03f,3.03466129e-03f,3.96241061e-03f,5.01316413e-03f,
    6.18666084e-03f,7.48261018e-03f,8.90069082e-03f,1.04405507e-02f,1.21018086e-02f,
    1.38840526e-02f,1.57868396e-02f,1.78096984e-02f,1.99521277e-02f,2.22135969e-02f,
    2.45935433e-02f,2.70913783e-02f,2.97064837e-02f,3.24382074e-02f,3.52858752e-02f,
    3.82487774e-02f,4.13261838e-02f,4.45173271e-02f,4.78214212e-02f,5.12376390e-02f,
    5.47651425e-02f,5.84030487e-02f,6.21504597e-02f,6.60064444e-02f,6.99700564e-02f,
    7.40402937e-02f,7.82161653e-02f,8.24966207e-02f,8.68806094e-02f,9.13670436e-02f,
    9.59548056e-02f,1.00642763e-01f,1.05429746e-01f,1.10314570e-01f,1.15296029e-01f,
    1.20372884e-01f,1.25543877e-01f,1.30807728e-01f,1.36163130e-01f,1.41608745e-01f,
    1.47143230e-01f,1.52765229e-01f,1.58473313e-01f,1.64266109e-01f,1.70142144e-01f,
    1.76099971e-01f,1.82138130e-01f,1.88255101e-01f,1.94449380e-01f,2.00719416e-01f,
    2.07063675e-01f,2.13480577e-01f,2.19968528e-01f,2.26525918e-01f,2.33151123e-01f,
    2.39842504e-01f,2.46598393e-01f,2.53417104e-01f,2.60296971e-01f,2.67236292e-01f,
    2.74233311e-01f,2.81286329e-01f,2.88393587e-01f,2.95553297e-01f,3.02763730e-01f,
    3.10023040e-01f,3.17329496e-01f,3.24681222e-01f,3.32076430e-01f,3.39513272e-01f,
    3.46989930e-01f,3.54504526e-01f,3.62055182e-01f,3.69640052e-01f,3.77257258e-01f,
    3.84904891e-01f,3.92581075e-01f,4.00283873e-01f,4.08011436e-01f,4.15761769e-01f,
    4.23533022e-01f,4.31323230e-01f,4.39130455e-01f,4.46952790e-01f,4.54788268e-01f,
    4.62634951e-01f,4.70490903e-01f,4.78354186e-01f,4.86222833e-01f,4.94094878e-01f,
    5.01968384e-01f,5.09841442e-01f,5.17712057e-01f,5.25578260e-01f,5.33438087e-01f,
    5.41289687e-01f,5.49130976e-01f,5.56960166e-01f,5.64775169e-01f,5.72574139e-01f,
    5.80355108e-01f,5.88116109e-01f,5.95855296e-01f,6.03570759e-01f,6.11260474e-01f,
    6.18922591e-01f,6.26555264e-01f,6.34156525e-01f,6.41724527e-01f,6.49257421e-01f,
    6.56753242e-01f,6.64210260e-01f,6.71626508e-01f,6.79000199e-01f,6.86329484e-01f,
    6.93612635e-01f,7.00847685e-01f,7.08033025e-01f,7.15166688e-01f,7.22247064e-01f,
    7.29272306e-01f,7.36240685e-01f,7.43150473e-01f,7.50000000e-01f,7.56787539e-01f,
    7.63511360e-01f,7.70169854e-01f,7.76761353e-01f,7.83284247e-01f,7.89736867e-01f,
    7.96117604e-01f,8.02424967e-01f,8.08657348e-01f,8.14813137e-01f,8.20890903e-01f,
    8.26889098e-01f,8.32806170e-01f,8.38640809e-01f,8.44391406e-01f,8.50056648e-01f,
    8.55635047e-01f,8.61125231e-01f,8.66525948e-01f,8.71835709e-01f,8.77053320e-01f,
    8.82177413e-01f,8.87206674e-01f,8.92139971e-01f,8.96976054e-01f,9.01713669e-01f,
    9.06351686e-01f,9.10888910e-01f,9.15324271e-01f,9.19656634e-01f,9.23884928e-01f,
    9.28008080e-01f,9.32025135e-01f,9.35935080e-01f,9.39736903e-01f,9.43429649e-01f,
    9.47012484e-01f,9.50484455e-01f,9.53844666e-01f,9.57092404e-01f,9.60226774e-01f,
    9.63247061e-01f,9.66152430e-01f,9.68942165e-01f,9.71615672e-01f,9.74172235e-01f,
    9.76611197e-01f,9.78931963e-01f,9.81133997e-01f,9.83216703e-01f,9.85179603e-01f,
    9.87022161e-01f,9.88743961e-01f,9.90344584e-01f,9.91823614e-01f,9.93180633e-01f,
    9.94415402e-01f,9.95527565e-01f,9.96516883e-01f,9.97382998e-01f,9.98125851e-01f,
    9.98745143e-01f,9.99240756e-01f,9.99612570e-01f,9.99860525e-01f,9.99984503e-01f,
    9.99984503e-01f,9.99860525e-01f,9.99612570e-01f,9.99240756e-01f,9.98745143e-01f,
    9.98125851e-01f,9.97382998e-01f,9.96516883e-01f,9.95527565e-01f,9.94415402e-01f,
    9.93180633e-01f,9.91823614e-01f,9.90344584e-01f,9.88743961e-01f,9.87022161e-01f,
    9.85179603e-01f,9.83216703e-01f,9.81133997e-01f,9.78931963e-01f,9.76611197e-01f,
    9.74172235e-01f,9.71615672e-01f,9.68942165e-01f,9.66152430e-01f,9.63247061e-01f,
    9.60226774e-01f,9.57092404e-01f,9.53844666e-01f,9.50484455e-01f,9.47012484e-01f,
    9.43429649e-01f,9.39736903e-01f,9.35935080e-01f,9.32025135e-01f,9.28008080e-01f,
    9.23884928e-01f,9.19656634e-01f,9.15324271e-01f,9.10888910e-01f,9.06351686e-01f,
    9.01713669e-01f,8.96976054e-01f,8.92139971e-01f,8.87206674e-01f,8.82177413e-01f,
    8.77053320e-01f,8.71835709e-01f,8.66525948e-01f,8.61125231e-01f,8.55635047e-01f,
    8.50056648e-01f,8.44391406e-01f,8.38640809e-01f,8.32806170e-01f,8.26889098e-01f,
    8.20890903e-01f,8.14813137e-01f,8.08657348e-01f,8.02424967e-01f,7.96117604e-01f,
    7.89736867e-01f,7.83284247e-01f,7.76761353e-01f,7.70169854e-01f,7.63511360e-01f,
    7.56787539e-01f,7.50000000e-01f,7.43150473e-01f,7.36240685e-01f,7.29272306e-01f,
    7.22247064e-01f,7.15166688e-01f,7.08033025e-01f,7.00847685e-01f,6.93612635e-01f,
    6.86329484e-01f,6.79000199e-01f,6.71626508e-01f,6.64210260e-01f,6.56753242e-01f,
    6.49257421e-01f,6.41724527e-01f,6.34156525e-01f,6.26555264e-01f,6.18922591e-01f,
    6.11260474e-01f,6.03570759e-01f,5.95855296e-01f,5.88116109e-01f,5.80355108e-01f,
    5.72574139e-01f,5.64775169e-01f,5.56960166e-01f,5.49130976e-01f,5.41289687e-01f,
    5.33438087e-01f,5.25578260e-01f,5.17712057e-01f,5.09841442e-01f,5.01968384e-01f,
    4.94094878e-01f,4.86222833e-01f,4.78354186e-01f,4.70490903e-01f,4.62634951e-01f,
    4.54788268e-01f,4.46952790e-01f,4.39130455e-01f,4.31323230e-01f,4.23533022e-01f,
    4.15761769e-01f,4.08011436e-01f,4.00283873e-01f,3.92581075e-01f,3.84904891e-01f,
    3.77257258e-01f,3.69640052e-01f,3.62055182e-01f,3.54504526e-01f,3.46989930e-01f,
    3.39513272e-01f,3.32076430e-01f,3.24681222e-01f,3.17329496e-01f,3.10023040e-01f,
    3.02763730e-01f,2.95553297e-01f,2.88393587e-01f,2.81286329e-01f,2.74233311e-01f,
    2.67236292e-01f,2.60296971e-01f,2.53417104e-01f,2.46598393e-01f,2.39842504e-01f,
    2.33151123e-01f,2.26525918e-01f,2.19968528e-01f,2.13480577e-01f,2.07063675e-01f,
    2.00719416e-01f,1.94449380e-01f,1.88255101e-01f,1.82138130e-01f,1.76099971e-01f,
    1.70142144e-01f,1.64266109e-01f,1.58473313e-01f,1.52765229e-01f,1.47143230e-01f,
    1.41608745e-01f,1.36163130e-01f,1.30807728e-01f,1.25543877e-01f,1.20372884e-01f,
    1.15296029e-01f,1.10314570e-01f,1.05429746e-01f,1.00642763e-01f,9.59548056e-02f,
    9.13670436e-02f,8.68806094e-02f,8.24966207e-02f,7.82161653e-02f,7.40402937e-02f,
    6.99700564e-02f,6.60064444e-02f,6.21504597e-02f,5.84030487e-02f,5.47651425e-02f,
    5.12376390e-02f,4.78214212e-02f,4.45173271e-02f,4.13261838e-02f,3.82487774e-02f,
    3.52858752e-02f,3.24382074e-02f,2.97064837e-02f,2.70913783e-02f,2.45935433e-02f,
    2.22135969e-02f,1.99521277e-02f,1.78096984e-02f,1.57868396e-02f,1.38840526e-02f,
    1.21018086e-02f,1.04405507e-02f,8.90069082e-03f,7.48261018e-03f,6.18666084e-03f,
    5.01316413e-03f,3.96241061e-03f,3.03466129e-03f,2.23014620e-03f,1.54906476e-03f,
    9.91585897e-04f,5.57847728e-04f,2.47957942e-04f,6.19933344e-05f,1.49975976e-32f,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0
};

struct MelSpec {
    struct MelEntry { int mel_bin; float weight; };
    std::vector<std::vector<MelEntry>> sparse_fb;  // [N_FREQ]

    cufftHandle fft_plan = 0;
    int fft_plan_batch = 0;
    float* d_frames = nullptr;
    cufftComplex* d_fft_out = nullptr;
    size_t frames_cap = 0, fft_cap = 0;

    void init() {
        sparse_fb.resize(N_FREQ);
        for (int i = 0; i < N_MEL_ENTRIES; i++)
            sparse_fb[MEL_FILTERBANK[i].freq].push_back(
                {MEL_FILTERBANK[i].mel, MEL_FILTERBANK[i].weight});
    }

    ~MelSpec() {
        if (fft_plan) cufftDestroy(fft_plan);
        if (d_frames) cudaFree(d_frames);
        if (d_fft_out) cudaFree(d_fft_out);
    }

    void ensure_fft(int n_frames) {
        size_t need_frames = (size_t)n_frames * N_FFT * sizeof(float);
        if (need_frames > frames_cap) {
            if (d_frames) cudaFree(d_frames);
            CUDA_CHECK(cudaMalloc(&d_frames, need_frames));
            frames_cap = need_frames;
        }
        size_t need_fft = (size_t)n_frames * N_FREQ * sizeof(cufftComplex);
        if (need_fft > fft_cap) {
            if (d_fft_out) cudaFree(d_fft_out);
            CUDA_CHECK(cudaMalloc(&d_fft_out, need_fft));
            fft_cap = need_fft;
        }
        if (n_frames != fft_plan_batch) {
            if (fft_plan) cufftDestroy(fft_plan);
            int n[] = {N_FFT};
            cufftResult rc = cufftPlanMany(&fft_plan, 1, n,
                n, 1, N_FFT,
                n, 1, N_FREQ,
                CUFFT_R2C, n_frames);
            if (rc != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT plan failed: %d\n", rc); std::exit(1);
            }
            fft_plan_batch = n_frames;
        }
    }

    void compute(const float* audio, int num_samples,
                 std::vector<float>& features, int& n_frames, int& n_valid) {
        std::vector<float> preemph(num_samples);
        preemph[0] = audio[0];
        for (int i = 1; i < num_samples; i++)
            preemph[i] = audio[i] - PREEMPH * audio[i - 1];

        int pad = N_FFT / 2;
        int padded_len = num_samples + 2 * pad;
        n_frames = (padded_len - N_FFT) / HOP + 1;
        n_valid = num_samples / HOP;

        std::vector<float> frames(n_frames * N_FFT, 0.0f);
        for (int f = 0; f < n_frames; f++) {
            float* row = frames.data() + f * N_FFT;
            for (int i = 0; i < N_FFT; i++) {
                int pos = f * HOP + i - pad;
                float sample = (pos >= 0 && pos < num_samples) ? preemph[pos] : 0.0f;
                row[i] = sample * HANN_WINDOW[i];
            }
        }

        ensure_fft(n_frames);
        CUDA_CHECK(cudaMemcpy(d_frames, frames.data(),
                               n_frames * N_FFT * sizeof(float),
                               cudaMemcpyHostToDevice));
        cufftResult rc = cufftExecR2C(fft_plan, d_frames, d_fft_out);
        if (rc != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT exec failed: %d\n", rc); std::exit(1);
        }
        std::vector<cufftComplex> fft_out(n_frames * N_FREQ);
        CUDA_CHECK(cudaMemcpy(fft_out.data(), d_fft_out,
                               n_frames * N_FREQ * sizeof(cufftComplex),
                               cudaMemcpyDeviceToHost));

        std::vector<float> power(n_frames * N_FREQ);
        for (int i = 0; i < n_frames * N_FREQ; i++)
            power[i] = fft_out[i].x * fft_out[i].x + fft_out[i].y * fft_out[i].y;

        std::vector<float> mel(n_frames * N_MELS, 0.0f);
        for (int f = 0; f < n_frames; f++) {
            float* mel_row = mel.data() + f * N_MELS;
            const float* pow_row = power.data() + f * N_FREQ;
            for (int k = 0; k < N_FREQ; k++) {
                float p = pow_row[k];
                for (auto& e : sparse_fb[k])
                    mel_row[e.mel_bin] += p * e.weight;
            }
        }

        for (auto& v : mel) v = logf(v + LOG_EPS);

        features.resize(N_MELS * n_valid);
        for (int m = 0; m < N_MELS; m++) {
            double sum = 0;
            for (int f = 0; f < n_valid; f++)
                sum += mel[f * N_MELS + m];
            float mean = (float)(sum / n_valid);

            double sq_sum = 0;
            for (int f = 0; f < n_valid; f++) {
                float d = mel[f * N_MELS + m] - mean;
                sq_sum += d * d;
            }
            float std_val = sqrtf((float)(sq_sum / (n_valid - 1))) + NORM_EPS;

            for (int f = 0; f < n_valid; f++)
                features[m * n_valid + f] = (mel[f * N_MELS + m] - mean) / std_val;
        }
    }
};

// ---------------------------------------------------------------------------
// Vocabulary (1025 BPE tokens, blank=1024)
// ---------------------------------------------------------------------------

static const char* const VOCAB[] = {
    "<unk>"," t"," th"," a","in"," the","re"," w"," o"," s","at","ou","er","nd"," i"," b",
    " c","on"," h","ing"," to"," m","en"," f"," p","an"," d","es","or","ll"," of"," and",
    " y"," l"," I","it"," in","is","ed"," g"," you","ar"," that","om","as"," n","ve",
    "us","ic","ow","al"," it"," be"," wh","le","ion","ut","ot"," we"," is"," e","et",
    "ay"," re"," on"," T"," A"," ha","ent","ke","ct"," S","ig","ver"," Th","all","id",
    " for","ro"," he","se"," this","ld","ly"," go"," k"," st","st","ch"," li"," u","am",
    "ur","ce","ith","im"," so"," have"," do","ht","th"," an"," with","ad"," r","ir",
    " was"," as"," W"," are","ust","ally"," j"," se","ation","od","ere"," like"," not",
    " kn","ight"," B"," they"," And"," know","ome","op"," can"," or"," sh"," me","ill",
    "ant","ck"," what"," at"," ab","ould","ol"," So"," C","use","ter","il"," but",
    " just"," ne"," de","ra","ore"," there","ul","out"," con"," all"," The","ers"," H",
    " fr"," pro","ge","ea"," Y"," O"," M","pp"," com","ess"," ch"," al","est","ate","qu",
    " lo"," ex","very"," su","ain"," one","ca","art","ist","if","ive"," if","ink","nt",
    "ab"," about"," going"," v"," wor","um","ok"," your"," my","ind"," get","cause",
    " from"," don","ri","pe","un","ity"," up"," P"," out","ort"," L","ment","el"," N",
    " some","ich","and"," think","em","oug"," G","os"," D","res"," because"," by","ake",
    " int","ie"," us"," tr"," then","ack"," pl"," here"," pe","her"," will"," F",
    " which","ard"," right"," thing"," want","ies","ople"," It"," them","ame"," We",
    "our"," say"," R"," people"," see"," who","ast","ure","ect","ear"," tim"," E"," You",
    " would"," when","ven"," our","ci"," really"," more","ound","ose","ak"," co","ide",
    "ough"," had","so"," qu","eah"," were","ine"," act","ther"," these"," how"," now",
    " sa","ud"," Wh"," man","ous","one","pt","ff","ong"," has"," any"," very"," But",
    " look","iv","itt"," time"," mo"," ar","hing"," le"," work"," their","are"," his",
    "per","ions"," im"," ag"," J"," no"," en"," got","ag"," sp","ans","act"," te",
    " also","iz","ice"," That"," cl"," been"," way"," fe"," did","ple","ually"," other",
    " U","ite","age","omet","ber","reat","ree"," into","own"," tw"," part","alk",
    " where"," need"," every","pl"," ad","ry"," over","ble","ap","ue"," kind"," po",
    " back"," cont","iff"," somet"," pr","nder","ire"," good"," than","ace"," gu","ep",
    "og","ick","way"," lot"," un"," things"," In","ish","kay"," well"," could"," pre",
    " two","irst"," diff","ach","cc","ittle","int"," He"," those","ence","ip","ase",
    " him"," make"," little","ical"," gr"," year","ass"," thr","uch","ated"," This",
    " off"," res","ac","ance"," actually"," talk","ult","able","orm"," dis"," first",
    "ations"," something"," she","sel"," let","ord"," may","ia"," am"," her"," said",
    " bo","be","ount"," much"," per"," even"," differe","vel","ary"," app","ving",
    " comm"," imp","ys"," again","ress"," yeah"," down","ang"," mean","na","ens"," does",
    " fo"," comp"," ro"," bl","ody"," K"," through"," start","uct"," only"," bet",
    " under"," br"," take","ning"," bu"," use"," Ch","xt","co","ory","ild"," put",
    " call"," new","other","ting"," happ","ater"," inc","ition"," different"," should",
    "ade","ign","thing"," day","fore"," Yeah","ark","ile","ial"," come"," They"," being",
    " try","ious"," sc"," bit"," spe","ub","fe"," doing"," St","vers","av","ty","ian",
    "onna","red","wn"," ke","form","ors"," fl","fter","ail","ents"," gonna"," point",
    "ces"," There","self"," many"," If"," same"," sy"," quest"," most"," great"," What",
    " fu","ug"," show","we","ual","ons"," Be","ically"," ser"," rem"," ind"," pers"," V",
    "he"," str","ved"," still","ank"," rec"," wr","ought","day","ath"," end"," bas","ft",
    "erm","body","ph","ject","ict"," play"," Is","ates"," ph","oth"," acc","get",
    " years"," em"," id"," Oh","ves","ever"," inter"," rel"," before"," feel","igh",
    " three","iss"," des","ne"," why"," uh"," To"," cons"," hel"," after","ower","urn",
    " okay"," long"," bel"," around","ful","te","ise"," ob"," supp","ady","ange","aking",
    " pos","atch"," tra","gr"," might","ert"," help","ost"," too","cial"," world",
    " give","ike"," Okay","ways"," min","ward","ily"," gen"," find"," dec","ular","ob",
    " tell"," Now"," sm"," cour"," real","cess","nds"," big"," num","ction"," add",
    " set"," um","ood","ible"," own"," life","ities"," its"," God","pect"," didn","stem",
    "les","uc","ib","ating","olog"," person"," inv","ably"," sure"," reg","lic"," stu",
    " cr"," ev","ments"," another"," la"," last"," sub"," att"," op"," inst"," sl",
    " happen"," rep"," import","ific","ix"," made"," ear"," ac"," def","ute"," next",
    "ative"," form"," guys"," system","ew"," able","ied"," always","ren","erest"," As",
    " mod"," done","ings"," love","ism"," ask","old","ered"," trans"," count","ility",
    " high"," fin"," prob"," pol"," exam"," pres"," maybe","ell"," stud"," prod"," car",
    "ock"," used","oy","stand"," No"," mon","ks"," interest"," ent","ited"," sort",
    " For"," today","ics"," vide"," bec"," Well"," Al"," important"," such"," run",
    " keep"," fact","ata","ss"," never","ween"," stuff","ract"," question","als"," sim",
    "vern","ather"," course"," Of","oc","ness","arch","ize"," All","ense","blem",
    " probably","hip"," number","ention"," saying"," commun"," An","akes"," belie",
    " between"," better","cus"," place"," gener"," ca"," ins"," ass","cond","cept","ull",
    " understand"," fun"," thought","gan","iew","cy","ution","ope","ason"," problem",
    " doesn","ational"," read"," trying"," sch"," el","ah","atter"," exper"," four",
    " ele"," cou","ont"," called"," partic"," open"," gl"," everything"," eff",
    " getting"," ty"," Am"," Because","ave"," met"," Like","oney"," ","e","t","o","a",
    "n","i","s","h","r","l","d","u","c","y","m","g","w","f","p",",","b",".","k","v","'",
    "I","T","A","S","j","x","W","B","C","?","0","O","-","M","H","Y","q","1","P","z","L",
    "D","N","G","F","R","E","2","J","U",":","5","9","3","K","4","V","8","6","7","!","%",
    "Q","$","Z","X","\xc3\xa9","/","\xc3\xad","\xc3\xa1","\xc2\xa3","\xc3\xb3",
    "\xc4\x81","\xc3\xbc","\xc3\xb1","\xc3\xb6","\xc3\xa8","\xc3\xa7","\xc3\xa0",
    "\xc2\xbf","\xce\xbc","\xcf\x80","\xc3\xa4","\xc3\xba","\xce\xb8","\xc3\xa3",
    "\xcf\x86","\xc4\xab","\xcf\x83","\xc3\xaa","\xcf\x81","\xc3\xa2","\xc3\xb4",
    "^","\xe2\x82\xac","\xc3\x89","\xc5\xab","\xce\x94","\xce\xbb","\xce\xb1",
    "\xcf\x84","\xc3\xa6","\xd0\xb0","\xd0\xbe","\xce\xbd","\xc3\xae","\xce\xb3",
    "\xcf\x88","\xc4\x93","\xd1\x82","\xc3\x9f","\xcf\x89","\xc3\xaf","\xc4\x87",
    "\xc4\x8d","\xce\xb5","\xd0\xb5","\xd0\xb8","\xc3\xb2","\xd1\x80","\xce\xb2",
    "\xc3\xb8","\xc5\x82","\xce\xb4","\xce\xb7","\xd0\xbf","\xc3\xab","\xd0\xbd",
    "\xd1\x81","\xc5\xa1","\xc3\x9c","\xc3\xa5","\xc5\x84","\xc5\x9b","\xd1\x8f",
    "\xc4\x91","\xd0\xbb","\xd0\xbc","\xc3\x96","\xc3\xbb","\xc8\x99","\xd0\xb2",
    "\xc3\x81","\xc3\x98","\xc3\xb9","\xce\xbf","\xd1\x87","\xd1\x8c","\xc5\xbe",
    "\xce\xa6","\xd1\x83","\xc4\x99","\xce\xb9","\xd0\xb1","\xd0\xb3","\xd0\xba",
    "\xc5\x91","\xc5\x9a","\xce\xa9","\xce\xba","\xcf\x85","\xc3\xac","\xc4\x8c",
    "\xce\xad","\xd1\x85","\xd1\x8b","\xc3\x85","\xc3\x87","\xc5\xbc","\xce\xaf",
    "\xce\xb6","\xcf\x87","\xd1\x8d","\xc3\x86","\xc3\x8d","\xc3\xb5","\xc4\x9b",
    "\xc4\xa7","\xc5\x81","\xc5\x93","\xc5\xbd","\xc8\x9b","\xce\x93","\xd0\x9f",
    "\xd0\xb4","\xd0\xb7","\xd1\x84","\xc2\xa1","\xc3\x80","\xc3\x8e","\xc4\x80",
    "\xc4\x97","\xc5\xa0","\xc5\xba","\xce\x9a","\xce\xa8","\xce\xac","\xce\xbe",
    "\xce\xbf","<blk>"
};
static constexpr int VOCAB_SIZE = 1025;
static constexpr int BLANK_ID = 1024;

static std::string detokenize(const std::vector<int>& ids) {
    std::string text;
    for (int id : ids)
        if (id >= 0 && id < VOCAB_SIZE)
            text += VOCAB[id];
    size_t start = text.find_first_not_of(' ');
    return (start == std::string::npos) ? "" : text.substr(start);
}

// ---------------------------------------------------------------------------
// Inference pipeline (CUDA backend)
// ---------------------------------------------------------------------------

struct Pipeline {
    Weights weights;
    CudaModel cuda_model;
    MelSpec mel;
    cudaStream_t stream = nullptr;

    void init(const std::string& weights_path) {
        using clk = std::chrono::high_resolution_clock;
        auto t0 = clk::now();

        CUDA_CHECK(cudaStreamCreate(&stream));
        weights = Weights::load(weights_path, stream);
        auto t_weights = clk::now();

        mel.init();
        mel.ensure_fft(1000);  // Pre-create cuFFT plan for ~10s audio
        auto t_mel = clk::now();

        cuda_model.init(weights, stream, 16000 * 120 / 160);  // 120s max audio
        auto t_model = clk::now();

        auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
        fprintf(stderr, "init: %.0fms (weights=%.0f mel=%.0f model=%.0f)\n",
                ms(t0,t_model), ms(t0,t_weights), ms(t_weights,t_mel), ms(t_mel,t_model));
    }

    ~Pipeline() {
        cuda_model.free();
        weights.free();
        if (stream) cudaStreamDestroy(stream);
    }

    double last_mel_ms = 0, last_enc_ms = 0, last_dec_ms = 0;

    std::string transcribe(const float* samples, int num_samples) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // --- 1. Mel spectrogram (CPU) ---
        std::vector<float> features;
        int n_frames, n_valid;
        mel.compute(samples, num_samples, features, n_frames, n_valid);
        auto t_mel = std::chrono::high_resolution_clock::now();

        // --- 2. Upload mel to GPU and run encoder ---
        GpuBuf d_features(N_MELS * n_valid * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(d_features.ptr, features.data(),
                                    N_MELS * n_valid * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int T = cuda_model.encode((const float*)d_features.ptr, n_valid);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t_enc = std::chrono::high_resolution_clock::now();

        // --- 3. TDT greedy decode ---
        auto text = decode(T);
        auto t_dec = std::chrono::high_resolution_clock::now();

        last_mel_ms = std::chrono::duration<double, std::milli>(t_mel - t_start).count();
        last_enc_ms = std::chrono::duration<double, std::milli>(t_enc - t_mel).count();
        last_dec_ms = std::chrono::duration<double, std::milli>(t_dec - t_enc).count();
        return text;
    }

private:
    std::string decode(int enc_len) {
        cuda_model.decoder_reset();

        std::vector<int> tokens;
        int last_token = BLANK_ID;
        int t = 0, emitted = 0;

        // Host buffer for joint output (FP16 → FP32 on CPU)
        std::vector<half> joint_host(D_OUTPUT);
        float logits[D_OUTPUT];

        while (t < enc_len) {
            half* joint_out = cuda_model.decode_step(t, last_token);

            CUDA_CHECK(cudaMemcpyAsync(joint_host.data(), joint_out,
                                        D_OUTPUT * sizeof(half),
                                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Convert FP16 → FP32
            for (int i = 0; i < D_OUTPUT; i++)
                logits[i] = __half2float(joint_host[i]);

            // Token = argmax(logits[:vocab_size])
            int token = 0;
            float best = logits[0];
            for (int i = 1; i < VOCAB_SIZE; i++)
                if (logits[i] > best) { best = logits[i]; token = i; }

            // Duration step = argmax(logits[vocab_size:])
            int step = 0;
            float best_dur = logits[VOCAB_SIZE];
            for (int i = 1; i < D_OUTPUT - VOCAB_SIZE; i++)
                if (logits[VOCAB_SIZE + i] > best_dur) {
                    best_dur = logits[VOCAB_SIZE + i]; step = i;
                }

            if (token != BLANK_ID) {
                cuda_model.decoder_commit();
                tokens.push_back(token);
                last_token = token;
                emitted++;
            }

            if (step > 0) { t += step; emitted = 0; }
            else if (token == BLANK_ID || emitted >= 10) { t++; emitted = 0; }
        }

        return detokenize(tokens);
    }
};

// ---------------------------------------------------------------------------
// HTTP server mode
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

static thread_local std::string t_log_detail;

static void log_request(const httplib::Request& req, const httplib::Response& res) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&tt, &tm);
    char ts[20];
    strftime(ts, sizeof(ts), "%H:%M:%S", &tm);

    fprintf(stderr, "%s  %s %s  %d\n", ts, req.method.c_str(), req.path.c_str(), res.status);

    if (!t_log_detail.empty()) {
        fprintf(stderr, "         %s\n", t_log_detail.c_str());
        t_log_detail.clear();
    }
}

static void run_server(Pipeline& pipeline, const std::string& host, int port) {
    httplib::Server svr;
    std::mutex mtx;

    svr.set_logger(log_request);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/transcribe", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'file' field\"}", "application/json");
            return;
        }
        const auto& file = req.get_file_value("file");
        auto wav = read_wav_from_memory(file.content.data(), file.content.size());
        if (wav.samples.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid WAV (need 16kHz mono, int16/float32)\"}", "application/json");
            return;
        }

        double audio_dur = (double)wav.samples.size() / wav.sample_rate;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::string text;
        {
            std::lock_guard<std::mutex> lock(mtx);
            text = pipeline.transcribe(wav.samples.data(), wav.samples.size());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::string preview = text.substr(0, 80);
        if (text.size() > 80) preview += "...";
        char detail[256];
        snprintf(detail, sizeof(detail), "audio=%.1fs  inference=%.0fms  RTFx=%.0fx  \"%s\"",
                 audio_dur, elapsed * 1000, audio_dur / elapsed, preview.c_str());
        t_log_detail = detail;

        char body[4096];
        snprintf(body, sizeof(body),
                 "{\"text\":\"%s\",\"audio_duration_s\":%.2f,\"inference_time_s\":%.4f}",
                 json_escape(text).c_str(), audio_dur, elapsed);
        res.set_content(body, "application/json");
    });

    const char* display_host = (host == "0.0.0.0") ? "localhost" : host.c_str();
    fprintf(stderr, "listening on http://%s:%d\n", display_host, port);
    fprintf(stderr, "\n");
    if (!svr.listen(host, port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--weights FILE] [--server [[host]:port]] <wav_file>...\n", argv[0]);
        return 1;
    }

    std::string weights_path = "weights.bin";
    std::vector<std::string> wav_files;
    bool server_mode = false;
    std::string server_host = "0.0.0.0";
    int server_port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--server") {
            server_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string addr = argv[++i];
                auto colon = addr.rfind(':');
                if (colon != std::string::npos) {
                    if (colon > 0) server_host = addr.substr(0, colon);
                    server_port = std::stoi(addr.substr(colon + 1));
                }
            }
        } else {
            wav_files.push_back(arg);
        }
    }

    if (!server_mode && wav_files.empty()) {
        fprintf(stderr, "No WAV files specified.\n");
        return 1;
    }

    using clk = std::chrono::high_resolution_clock;
    auto t_main_start = clk::now();

    // Force CUDA driver/context initialization
    cudaFree(0);
    auto t_cuda_init = clk::now();

    if (!server_mode) {
        int wav_fd = open(wav_files[0].c_str(), O_RDONLY);
        if (wav_fd >= 0) { posix_fadvise(wav_fd, 0, 0, POSIX_FADV_WILLNEED); close(wav_fd); }
    }

    Pipeline pipeline;
    pipeline.init(weights_path);
    auto t_init_done = clk::now();

    if (server_mode) {
        // Warmup with 1s of silence
        std::vector<float> silence(16000, 0.0f);
        pipeline.transcribe(silence.data(), silence.size());
        auto t_warmup_done = clk::now();

        auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t vram_free, vram_total;
        cudaMemGetInfo(&vram_free, &vram_total);

        auto file_size_mb = [](const std::string& path) -> double {
            struct stat st;
            return (stat(path.c_str(), &st) == 0) ? st.st_size / (1024.0 * 1024.0) : 0;
        };
        double weights_mb = file_size_mb(weights_path);

        fprintf(stderr, "\n");
        fprintf(stderr, "model:     parakeet-tdt-0.6b-v2 (custom CUDA)\n");
        fprintf(stderr, "weights:   %s (%.0f MB)\n", weights_path.c_str(), weights_mb);
        fprintf(stderr, "device:    %s (compute %d.%d, %.0f MB VRAM, %.0f MB free)\n",
                prop.name, prop.major, prop.minor,
                vram_total / (1024.0 * 1024.0), vram_free / (1024.0 * 1024.0));
        fprintf(stderr, "startup:   %.0f ms (cuda_init=%.0f weights+model=%.0f warmup=%.0f)\n",
                ms(t_main_start, t_warmup_done), ms(t_main_start, t_cuda_init),
                ms(t_cuda_init, t_init_done), ms(t_init_done, t_warmup_done));
        fprintf(stderr, "endpoints: GET /health | POST /transcribe\n");
        fprintf(stderr, "\n");

        run_server(pipeline, server_host, server_port);
        return 0;
    }

    // CLI mode: warmup with first file
    auto warmup_wav = read_wav(wav_files[0]);
    auto t_wav_read = clk::now();
    pipeline.transcribe(warmup_wav.samples.data(), warmup_wav.samples.size());
    auto t_warmup_done = clk::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    fprintf(stderr, "startup: %.0fms total (cuda_init=%.0f init=%.0f wav_read=%.0f warmup=%.0f)\n",
            ms(t_main_start, t_warmup_done), ms(t_main_start, t_cuda_init),
            ms(t_cuda_init, t_init_done),
            ms(t_init_done, t_wav_read), ms(t_wav_read, t_warmup_done));

    // Process each file
    for (size_t fi = 0; fi < wav_files.size(); fi++) {
        WavData wav = (fi == 0) ? std::move(warmup_wav) : read_wav(wav_files[fi]);
        double audio_dur = (double)wav.samples.size() / wav.sample_rate;

        auto t0 = std::chrono::high_resolution_clock::now();
        std::string text = pipeline.transcribe(wav.samples.data(), wav.samples.size());
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        printf("%s\n", text.c_str());
        fprintf(stderr, "%.1fs audio, %.1fms (mel=%.1f enc=%.1f dec=%.1f), %.0fx RTFx\n",
                audio_dur, elapsed * 1000,
                pipeline.last_mel_ms, pipeline.last_enc_ms, pipeline.last_dec_ms,
                audio_dur / elapsed);
    }

    return 0;
}
