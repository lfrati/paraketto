// parakeet.cpp — Clean TensorRT C++ runtime for Parakeet TDT 0.6B V2
//
// Single-file inference: reads a WAV file, computes mel spectrogram
// (cuFFT for batched FFT, CPU for the rest), runs encoder and decoder
// TRT engines on GPU, prints transcription.
//
// Build: make parakeet
// Usage: ./parakeet audio.wav

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

#include <cuda_runtime.h>
#include <cufft.h>
#include "NvInfer.h"

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

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

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
// TensorRT logger + engine wrapper
// ---------------------------------------------------------------------------

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

static Logger gLogger;

struct Engine {
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime{nullptr, nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine{nullptr, nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context{nullptr, nullptr};

    static Engine load(const std::string& path) {
        Engine e;
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { fprintf(stderr, "Cannot open engine: %s\n", path.c_str()); std::exit(1); }
        size_t size = f.tellg(); f.seekg(0);
        std::vector<char> data(size);
        f.read(data.data(), size);
        e.runtime = {nvinfer1::createInferRuntime(gLogger), [](nvinfer1::IRuntime* r) { delete r; }};
        e.engine = {e.runtime->deserializeCudaEngine(data.data(), size), [](nvinfer1::ICudaEngine* eng) { delete eng; }};
        if (!e.engine) { fprintf(stderr, "Failed to deserialize: %s\n", path.c_str()); std::exit(1); }
        e.context = {e.engine->createExecutionContext(), [](nvinfer1::IExecutionContext* c) { delete c; }};
        return e;
    }

    void bind(const char* name, void* ptr) {
        if (!context->setTensorAddress(name, ptr)) {
            fprintf(stderr, "Failed to bind: %s\n", name); std::exit(1);
        }
    }
    void setShape(const char* name, nvinfer1::Dims dims) {
        if (!context->setInputShape(name, dims)) {
            fprintf(stderr, "Failed to set shape: %s\n", name); std::exit(1);
        }
    }
    bool enqueue(cudaStream_t stream) { return context->enqueueV3(stream); }
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

// ---------------------------------------------------------------------------
// Mel spectrogram (CPU) — matches NeMo nemo128.onnx preprocessor
// ---------------------------------------------------------------------------

static std::vector<float> load_binary_f32(const std::string& path, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); std::exit(1); }
    std::vector<float> data(expected);
    f.read((char*)data.data(), expected * sizeof(float));
    return data;
}

struct MelSpec {
    std::vector<float> hann;          // [N_FFT]
    // Sparse mel filterbank: for each freq bin k, list of (mel_bin, weight) pairs
    struct MelEntry { int mel_bin; float weight; };
    std::vector<std::vector<MelEntry>> sparse_fb;  // [N_FREQ]

    // cuFFT plan (cached for a given batch size)
    cufftHandle fft_plan = 0;
    int fft_plan_batch = 0;
    // GPU buffers for cuFFT (reused across calls, grow-only)
    float* d_frames = nullptr;            // [n_frames, N_FFT] real input
    cufftComplex* d_fft_out = nullptr;    // [n_frames, N_FREQ] complex output
    size_t frames_cap = 0, fft_cap = 0;

    void init(const std::string& model_dir) {
        hann = load_binary_f32(model_dir + "/hann_window.bin", N_FFT);
        auto mel_fb = load_binary_f32(model_dir + "/mel_filterbank.bin", N_FREQ * N_MELS);
        // Build sparse filterbank (98.5% sparse — each freq hits at most 2 mel bins)
        sparse_fb.resize(N_FREQ);
        for (int k = 0; k < N_FREQ; k++)
            for (int m = 0; m < N_MELS; m++)
                if (mel_fb[k * N_MELS + m] != 0.0f)
                    sparse_fb[k].push_back({m, mel_fb[k * N_MELS + m]});
    }

    ~MelSpec() {
        if (fft_plan) cufftDestroy(fft_plan);
        if (d_frames) cudaFree(d_frames);
        if (d_fft_out) cudaFree(d_fft_out);
    }

    // Ensure cuFFT plan and GPU buffers are large enough for n_frames
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
                n, 1, N_FFT,    // inembed, istride, idist
                n, 1, N_FREQ,   // onembed, ostride, odist
                CUFFT_R2C, n_frames);
            if (rc != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT plan failed: %d\n", rc); std::exit(1);
            }
            fft_plan_batch = n_frames;
        }
    }

    // Compute mel spectrogram features. Output: [N_MELS, n_valid] (channel-first)
    void compute(const float* audio, int num_samples,
                 std::vector<float>& features, int& n_frames, int& n_valid) {
        // 1. Pre-emphasis
        std::vector<float> preemph(num_samples);
        preemph[0] = audio[0];
        for (int i = 1; i < num_samples; i++)
            preemph[i] = audio[i] - PREEMPH * audio[i - 1];

        // 2. Frame count
        int pad = N_FFT / 2;
        int padded_len = num_samples + 2 * pad;
        n_frames = (padded_len - N_FFT) / HOP + 1;
        n_valid = num_samples / HOP;

        // 3. Extract windowed frames into contiguous buffer (CPU)
        //    frames[f][i] = padded[f*HOP + i] * hann[i], with zero-padding
        std::vector<float> frames(n_frames * N_FFT, 0.0f);
        for (int f = 0; f < n_frames; f++) {
            float* row = frames.data() + f * N_FFT;
            for (int i = 0; i < N_FFT; i++) {
                int pos = f * HOP + i - pad;
                float sample = (pos >= 0 && pos < num_samples) ? preemph[pos] : 0.0f;
                row[i] = sample * hann[i];
            }
        }

        // 4. Batched R2C FFT on GPU via cuFFT
        ensure_fft(n_frames);
        CUDA_CHECK(cudaMemcpy(d_frames, frames.data(),
                               n_frames * N_FFT * sizeof(float),
                               cudaMemcpyHostToDevice));
        cufftResult rc = cufftExecR2C(fft_plan, d_frames, d_fft_out);
        if (rc != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT exec failed: %d\n", rc); std::exit(1);
        }
        // Download complex output
        std::vector<cufftComplex> fft_out(n_frames * N_FREQ);
        CUDA_CHECK(cudaMemcpy(fft_out.data(), d_fft_out,
                               n_frames * N_FREQ * sizeof(cufftComplex),
                               cudaMemcpyDeviceToHost));

        // 5. Power spectrum (CPU)
        std::vector<float> power(n_frames * N_FREQ);
        for (int i = 0; i < n_frames * N_FREQ; i++)
            power[i] = fft_out[i].x * fft_out[i].x + fft_out[i].y * fft_out[i].y;

        // 6. Sparse mel filterbank: power [n_frames, 257] → mel [n_frames, 128]
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

        // 7. Log
        for (auto& v : mel) v = logf(v + LOG_EPS);

        // 8. Transpose to [128, n_valid] and normalize per channel
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
// Vocabulary
// ---------------------------------------------------------------------------

struct Vocab {
    std::vector<std::string> tokens;
    int blank_id = -1;
    int vocab_size = 0;

    static Vocab load(const std::string& path) {
        Vocab v;
        std::ifstream f(path);
        if (!f) { fprintf(stderr, "Cannot open vocab: %s\n", path.c_str()); std::exit(1); }
        std::string line;
        int max_id = -1;
        std::vector<std::pair<std::string, int>> entries;
        while (std::getline(f, line)) {
            auto sp = line.rfind(' ');
            if (sp == std::string::npos) continue;
            std::string token = line.substr(0, sp);
            int id = std::stoi(line.substr(sp + 1));
            entries.push_back({token, id});
            if (id > max_id) max_id = id;
        }
        v.tokens.resize(max_id + 1);
        for (auto& [token, id] : entries) {
            std::string clean;
            for (size_t i = 0; i < token.size(); ) {
                if (i + 2 < token.size() &&
                    (unsigned char)token[i] == 0xe2 &&
                    (unsigned char)token[i+1] == 0x96 &&
                    (unsigned char)token[i+2] == 0x81) {
                    clean += ' '; i += 3;
                } else {
                    clean += token[i]; i++;
                }
            }
            v.tokens[id] = clean;
            if (token == "<blk>") v.blank_id = id;
        }
        v.vocab_size = max_id + 1;
        assert(v.blank_id >= 0);
        return v;
    }
};

static std::string detokenize(const Vocab& vocab, const std::vector<int>& ids) {
    std::string text;
    for (int id : ids)
        if (id >= 0 && id < (int)vocab.tokens.size())
            text += vocab.tokens[id];
    size_t start = text.find_first_not_of(' ');
    return (start == std::string::npos) ? "" : text.substr(start);
}

// ---------------------------------------------------------------------------
// Inference pipeline
// ---------------------------------------------------------------------------

struct Pipeline {
    Engine encoder, decoder_joint;
    MelSpec mel;
    Vocab vocab;
    cudaStream_t stream = nullptr;
    // GPU buffers
    GpuBuf d_features, d_length, d_enc_out, d_enc_lens;
    GpuBuf d_enc_frame;       // [1, 1024, 1]
    GpuBuf d_targets, d_target_length;
    GpuBuf d_states1, d_states2, d_out_states1, d_out_states2;
    GpuBuf d_dec_out, d_prednet_lens;

    void init(const std::string& engine_dir, const std::string& model_dir) {
        encoder = Engine::load(engine_dir + "/encoder.engine");
        decoder_joint = Engine::load(engine_dir + "/decoder_joint.engine");
        mel.init(model_dir);
        vocab = Vocab::load(model_dir + "/vocab.txt");
        CUDA_CHECK(cudaStreamCreate(&stream));

        d_length       = GpuBuf(sizeof(int64_t));
        d_enc_lens     = GpuBuf(sizeof(int64_t));
        d_enc_frame    = GpuBuf(1024 * sizeof(float));
        d_targets      = GpuBuf(sizeof(int32_t));
        d_target_length = GpuBuf(sizeof(int32_t));
        d_states1      = GpuBuf(2 * 640 * sizeof(float));
        d_states2      = GpuBuf(2 * 640 * sizeof(float));
        d_out_states1  = GpuBuf(2 * 640 * sizeof(float));
        d_out_states2  = GpuBuf(2 * 640 * sizeof(float));
        d_dec_out      = GpuBuf(1030 * sizeof(float));
        d_prednet_lens = GpuBuf(sizeof(int32_t));
    }

    ~Pipeline() { if (stream) cudaStreamDestroy(stream); }

    double last_mel_ms = 0, last_enc_ms = 0, last_dec_ms = 0;

    std::string transcribe(const float* samples, int num_samples) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // --- 1. Mel spectrogram (CPU) ---
        std::vector<float> features;
        int n_frames, n_valid;
        mel.compute(samples, num_samples, features, n_frames, n_valid);

        auto t_mel = std::chrono::high_resolution_clock::now();

        // --- 2. Encoder (GPU) ---
        d_features.resize(N_MELS * n_valid * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(d_features.ptr, features.data(),
                                    N_MELS * n_valid * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

        int64_t mel_len = n_valid;
        CUDA_CHECK(cudaMemcpyAsync(d_length.ptr, &mel_len,
                                    sizeof(int64_t), cudaMemcpyHostToDevice, stream));

        encoder.setShape("audio_signal", nvinfer1::Dims3{1, N_MELS, n_valid});
        encoder.setShape("length", nvinfer1::Dims{1, {1}});
        encoder.bind("audio_signal", d_features.ptr);
        encoder.bind("length", d_length.ptr);

        int max_enc = (n_valid / 8) + 10;
        d_enc_out.resize(1024 * max_enc * sizeof(float));
        encoder.bind("outputs", d_enc_out.ptr);
        encoder.bind("encoded_lengths", d_enc_lens.ptr);

        encoder.enqueue(stream);

        int64_t enc_len = 0;
        CUDA_CHECK(cudaMemcpyAsync(&enc_len, d_enc_lens.ptr,
                                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (enc_len > max_enc) enc_len = max_enc;

        auto t_enc = std::chrono::high_resolution_clock::now();

        // --- 3. TDT greedy decode ---
        auto text = decode(enc_len);

        auto t_dec = std::chrono::high_resolution_clock::now();
        last_mel_ms = std::chrono::duration<double, std::milli>(t_mel - t_start).count();
        last_enc_ms = std::chrono::duration<double, std::milli>(t_enc - t_mel).count();
        last_dec_ms = std::chrono::duration<double, std::milli>(t_dec - t_enc).count();
        return text;
    }

private:
    std::string decode(int64_t enc_len) {
        decoder_joint.bind("encoder_outputs", d_enc_frame.ptr);
        decoder_joint.bind("targets", d_targets.ptr);
        decoder_joint.bind("target_length", d_target_length.ptr);
        decoder_joint.bind("input_states_1", d_states1.ptr);
        decoder_joint.bind("input_states_2", d_states2.ptr);
        decoder_joint.bind("outputs", d_dec_out.ptr);
        decoder_joint.bind("output_states_1", d_out_states1.ptr);
        decoder_joint.bind("output_states_2", d_out_states2.ptr);
        decoder_joint.bind("prednet_lengths", d_prednet_lens.ptr);

        CUDA_CHECK(cudaMemsetAsync(d_states1.ptr, 0, 2 * 640 * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_states2.ptr, 0, 2 * 640 * sizeof(float), stream));
        int32_t one = 1;
        CUDA_CHECK(cudaMemcpyAsync(d_target_length.ptr, &one,
                                    sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        std::vector<int> tokens;
        int last_token = vocab.blank_id;
        float dec_out_host[1030];
        int t = 0, emitted = 0;

        while (t < enc_len) {
            // Extract encoder frame t: gather from [1, 1024, T'] strided layout
            CUDA_CHECK(cudaMemcpy2DAsync(
                d_enc_frame.ptr, sizeof(float),
                (char*)d_enc_out.ptr + t * sizeof(float), enc_len * sizeof(float),
                sizeof(float), 1024,
                cudaMemcpyDeviceToDevice, stream));

            int32_t target = (int32_t)last_token;
            CUDA_CHECK(cudaMemcpyAsync(d_targets.ptr, &target,
                                        sizeof(int32_t), cudaMemcpyHostToDevice, stream));

            decoder_joint.enqueue(stream);

            CUDA_CHECK(cudaMemcpyAsync(dec_out_host, d_dec_out.ptr,
                                        1030 * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Token = argmax(logits[:vocab_size])
            int token = 0;
            float best = dec_out_host[0];
            for (int i = 1; i < vocab.vocab_size; i++)
                if (dec_out_host[i] > best) { best = dec_out_host[i]; token = i; }

            // Duration step = argmax(logits[vocab_size:])
            int step = 0;
            float best_dur = dec_out_host[vocab.vocab_size];
            for (int i = 1; i < 1030 - vocab.vocab_size; i++)
                if (dec_out_host[vocab.vocab_size + i] > best_dur) {
                    best_dur = dec_out_host[vocab.vocab_size + i]; step = i;
                }

            if (token != vocab.blank_id) {
                std::swap(d_states1, d_out_states1);
                std::swap(d_states2, d_out_states2);
                decoder_joint.bind("input_states_1", d_states1.ptr);
                decoder_joint.bind("input_states_2", d_states2.ptr);
                decoder_joint.bind("output_states_1", d_out_states1.ptr);
                decoder_joint.bind("output_states_2", d_out_states2.ptr);
                tokens.push_back(token);
                last_token = token;
                emitted++;
            }

            if (step > 0) { t += step; emitted = 0; }
            else if (token == vocab.blank_id || emitted >= 10) { t++; emitted = 0; }
        }

        return detokenize(vocab, tokens);
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--engine-dir DIR] [--model-dir DIR] <wav_file>...\n", argv[0]);
        return 1;
    }

    std::string engine_dir = "engines";
    std::string model_dir = "model";
    std::vector<std::string> wav_files;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--engine-dir" && i + 1 < argc)
            engine_dir = argv[++i];
        else if (std::string(argv[i]) == "--model-dir" && i + 1 < argc)
            model_dir = argv[++i];
        else
            wav_files.push_back(argv[i]);
    }

    if (wav_files.empty()) {
        fprintf(stderr, "No WAV files specified.\n");
        return 1;
    }

    Pipeline pipeline;
    pipeline.init(engine_dir, model_dir);

    // Warmup with first file
    auto warmup_wav = read_wav(wav_files[0]);
    pipeline.transcribe(warmup_wav.samples.data(), warmup_wav.samples.size());

    // Process each file
    for (auto& path : wav_files) {
        auto wav = read_wav(path);
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
