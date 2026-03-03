// paraketto.cpp — TensorRT C++ runtime for Parakeet TDT 0.6B V2
//
// Build: make paraketto
// Usage: ./paraketto audio.wav

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include "NvInfer.h"

#include "common.h"
#include "wav.h"
#include "mel.h"
#include "vocab.h"
#include "server.h"

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
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "Cannot open engine: %s\n", path.c_str()); std::exit(1); }
        struct stat st;
        fstat(fd, &st);
        size_t size = st.st_size;
        void* data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (data == MAP_FAILED) { fprintf(stderr, "mmap failed: %s\n", path.c_str()); close(fd); std::exit(1); }
        madvise(data, size, MADV_SEQUENTIAL);

        e.runtime = {nvinfer1::createInferRuntime(gLogger), [](nvinfer1::IRuntime* r) { delete r; }};
        e.engine = {e.runtime->deserializeCudaEngine(data, size), [](nvinfer1::ICudaEngine* eng) { delete eng; }};

        munmap(data, size);
        close(fd);
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
// Inference pipeline
// ---------------------------------------------------------------------------

struct Pipeline {
    Engine encoder, decoder_joint;
    MelSpec mel;
    cudaStream_t stream = nullptr;
    // GPU buffers
    GpuBuf d_features, d_length, d_enc_out, d_enc_lens;
    GpuBuf d_enc_frame;       // [1, 1024, 1]
    GpuBuf d_targets, d_target_length;
    GpuBuf d_states1, d_states2, d_out_states1, d_out_states2;
    GpuBuf d_dec_out, d_prednet_lens;

    void init(const std::string& engine_dir) {
        using clk = std::chrono::high_resolution_clock;
        auto t0 = clk::now();

        // Load both engines in parallel
        std::thread dec_thread([&]{ decoder_joint = Engine::load(engine_dir + "/decoder_joint.engine"); });
        encoder = Engine::load(engine_dir + "/encoder.engine");
        dec_thread.join();
        auto t_eng = clk::now();

        mel.init();
        CUDA_CHECK(cudaStreamCreate(&stream));
        auto t_mel = clk::now();

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
        auto t_buf = clk::now();

        auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
        fprintf(stderr, "init: %.0fms (engines=%.0f mel+stream=%.0f bufs=%.0f)\n",
                ms(t0,t_buf), ms(t0,t_eng), ms(t_eng,t_mel), ms(t_mel,t_buf));
    }

    ~Pipeline() { if (stream) cudaStreamDestroy(stream); }

    double last_mel_ms = 0, last_enc_ms = 0, last_dec_ms = 0;

    std::string transcribe(const float* samples, int num_samples) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // --- 1. Mel spectrogram (fused GPU pipeline) ---
        int n_frames, n_valid;
        d_features.resize(N_MELS * 12000 * sizeof(float));  // pre-size for max frames
        mel.compute(samples, num_samples, (float*)d_features.ptr, n_frames, n_valid, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto t_mel = std::chrono::high_resolution_clock::now();

        // --- 2. Encoder (GPU, mel already on device) ---

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
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Get actual output length from tensor shape (not encoded_lengths,
        // which can be off-by-one in FP16 engines for certain input lengths).
        auto out_dims = encoder.context->getTensorShape("outputs");
        int64_t enc_len = out_dims.d[out_dims.nbDims - 1];  // last dim = T'

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
        int last_token = BLANK_ID;
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
            for (int i = 1; i < VOCAB_SIZE; i++)
                if (dec_out_host[i] > best) { best = dec_out_host[i]; token = i; }

            // Duration step = argmax(logits[vocab_size:])
            int step = 0;
            float best_dur = dec_out_host[VOCAB_SIZE];
            for (int i = 1; i < 1030 - VOCAB_SIZE; i++)
                if (dec_out_host[VOCAB_SIZE + i] > best_dur) {
                    best_dur = dec_out_host[VOCAB_SIZE + i]; step = i;
                }

            if (token != BLANK_ID) {
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
            else if (token == BLANK_ID || emitted >= 10) { t++; emitted = 0; }
        }

        return detokenize(tokens);
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto usage = [&]() {
        fprintf(stderr,
            "Usage: %s [OPTIONS] <wav_file>...\n"
            "\n"
            "Speech-to-text using Paraketto (TensorRT backend).\n"
            "Accepts one or more 16kHz or 24kHz mono WAV files (int16 or float32).\n"
            "\n"
            "Options:\n"
            "  --engine-dir DIR           TensorRT engine directory [default: engines]\n"
            "  --server [[host]:port]     Start HTTP server [default: 0.0.0.0:8080]\n"
            "  -h, --help                 Show this help\n",
            argv[0]);
    };

    if (argc < 2) { usage(); return 1; }

    std::string engine_dir = "engines";
    std::vector<std::string> wav_files;
    bool server_mode = false;
    std::string server_host = "0.0.0.0";
    int server_port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        if (arg == "--engine-dir" && i + 1 < argc) {
            engine_dir = argv[++i];
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

    // Force CUDA driver/context initialization before anything else
    cudaFree(0);
    auto t_cuda_init = clk::now();

    if (!server_mode) {
        // Hint kernel to pre-cache the first WAV file while engines load
        int wav_fd = open(wav_files[0].c_str(), O_RDONLY);
        if (wav_fd >= 0) { posix_fadvise(wav_fd, 0, 0, POSIX_FADV_WILLNEED); close(wav_fd); }
    }

    Pipeline pipeline;
    pipeline.init(engine_dir);
    auto t_init_done = clk::now();

    if (server_mode) {
        // Warmup with 1s of silence
        std::vector<float> silence(16000, 0.0f);
        pipeline.transcribe(silence.data(), silence.size());
        auto t_warmup_done = clk::now();

        auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };

        // CUDA device info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t vram_free, vram_total;
        cudaMemGetInfo(&vram_free, &vram_total);

        // Engine file sizes
        auto file_size_mb = [](const std::string& path) -> double {
            struct stat st;
            return (stat(path.c_str(), &st) == 0) ? st.st_size / (1024.0 * 1024.0) : 0;
        };
        double enc_mb = file_size_mb(engine_dir + "/encoder.engine");
        double dec_mb = file_size_mb(engine_dir + "/decoder_joint.engine");

        fprintf(stderr, "\n");
        fprintf(stderr, "model:     parakeet-tdt-0.6b-v2\n");
        fprintf(stderr, "engines:   %s (encoder %.0f MB, decoder %.0f MB)\n",
                engine_dir.c_str(), enc_mb, dec_mb);
        fprintf(stderr, "device:    %s (compute %d.%d, %.0f MB VRAM, %.0f MB free)\n",
                prop.name, prop.major, prop.minor,
                vram_total / (1024.0 * 1024.0), vram_free / (1024.0 * 1024.0));
        fprintf(stderr, "startup:   %.0f ms (cuda_init=%.0f engines=%.0f warmup=%.0f)\n",
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
        // Reuse warmup WAV data for first file to avoid re-reading
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
