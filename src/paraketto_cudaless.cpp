// paraketto_cudaless.cpp — ASR inference via direct GPU ioctls (zero libcuda/libcudart)
//
// Build: make paraketto.cudaless
// Usage: ./paraketto.cudaless [--weights FILE] audio.wav

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cudaless_model.h"
#include "wav.h"
#include "vocab.h"
#include "server.h"

// ---------------------------------------------------------------------------
// Inference pipeline (cudaless backend)
// ---------------------------------------------------------------------------

struct Pipeline {
    CudalessWeights weights;
    CudalessModel model;
    GPU gpu;

    void init(CudalessWeights&& prefetched) {
        if (!gpu.init()) {
            fprintf(stderr, "GPU init failed\n");
            std::exit(1);
        }

        weights = std::move(prefetched);
        weights.upload(gpu);

        if (!model.init(gpu, weights, 16000 * 120 / 160)) {  // 120s max audio
            fprintf(stderr, "Model init failed\n");
            std::exit(1);
        }
    }

    double last_mel_ms = 0, last_enc_ms = 0, last_dec_ms = 0;

    std::string transcribe(const float* samples, int num_samples) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. Mel spectrogram
        int n_frames, n_valid;
        model.mel_compute(samples, num_samples, n_frames, n_valid);
        auto t_mel = std::chrono::high_resolution_clock::now();

        // 2. Encoder
        int T = model.encode_gpu(n_valid);
        auto t_enc = std::chrono::high_resolution_clock::now();

        // 3. TDT greedy decode
        auto text = decode(T);
        auto t_dec = std::chrono::high_resolution_clock::now();

        last_mel_ms = std::chrono::duration<double, std::milli>(t_mel - t_start).count();
        last_enc_ms = std::chrono::duration<double, std::milli>(t_enc - t_mel).count();
        last_dec_ms = std::chrono::duration<double, std::milli>(t_dec - t_enc).count();
        return text;
    }

private:
    std::string decode(int enc_len) {
        model.decoder_reset();

        std::vector<int> tokens;
        int last_token = BLANK_ID;
        int t = 0, emitted = 0;

        while (t < enc_len) {
            model.decode_step(t, last_token);

            // Read argmax result from CPU-accessible GPU memory
            int* argmax = (int*)model.argmax_out.cpu;
            int token = argmax[0];
            int step = argmax[1];

            if (token != BLANK_ID) {
                model.decoder_commit();
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

    // Prefetch weights in background while GPU initializes
    CudalessWeights prefetched;
    std::thread prefetch_thread([&]() {
        prefetched = CudalessWeights::prefetch(weights_path);
    });

    if (!server_mode) {
        int wav_fd = open(wav_files[0].c_str(), O_RDONLY);
        if (wav_fd >= 0) { posix_fadvise(wav_fd, 0, 0, POSIX_FADV_WILLNEED); close(wav_fd); }
    }

    prefetch_thread.join();
    auto t_prefetch = clk::now();

    Pipeline pipeline;
    pipeline.init(std::move(prefetched));
    auto t_init_done = clk::now();

    if (server_mode) {
        // Warmup with 1s of silence
        std::vector<float> silence(16000, 0.0f);
        pipeline.transcribe(silence.data(), silence.size());
        auto t_warmup_done = clk::now();

        auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };

        auto file_size_mb = [](const std::string& path) -> double {
            struct stat st;
            return (stat(path.c_str(), &st) == 0) ? st.st_size / (1024.0 * 1024.0) : 0;
        };
        double weights_mb = file_size_mb(weights_path);

        fprintf(stderr, "\n");
        fprintf(stderr, "model:     parakeet-tdt-0.6b-v2 (cudaless)\n");
        fprintf(stderr, "weights:   %s (%.0f MB)\n", weights_path.c_str(), weights_mb);
        fprintf(stderr, "startup:   %.0f ms (prefetch=%.0f load=%.0f warmup=%.0f)\n",
                ms(t_main_start, t_warmup_done),
                ms(t_main_start, t_prefetch),
                ms(t_prefetch, t_init_done), ms(t_init_done, t_warmup_done));
        fprintf(stderr, "endpoints: GET /health | POST /transcribe\n");
        fprintf(stderr, "\n");

        run_server(pipeline, server_host, server_port);
        return 0;
    }

    // CLI mode: warmup with first file
    auto warmup_wav = read_wav(wav_files[0]);
    pipeline.transcribe(warmup_wav.samples.data(), warmup_wav.samples.size());
    auto t_warmup_done = clk::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    fprintf(stderr, "startup: %.0fms (prefetch=%.0f load=%.0f warmup=%.0f)\n",
            ms(t_main_start, t_warmup_done),
            ms(t_main_start, t_prefetch),
            ms(t_prefetch, t_init_done),
            ms(t_init_done, t_warmup_done));

    // Process each file
    for (size_t fi = 0; fi < wav_files.size(); fi++) {
        WavData wav = (fi == 0) ? std::move(warmup_wav) : read_wav(wav_files[fi]);
        double audio_dur = (double)wav.samples.size() / wav.sample_rate;

        auto t0 = clk::now();
        std::string text = pipeline.transcribe(wav.samples.data(), wav.samples.size());
        auto t1 = clk::now();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        printf("%s\n", text.c_str());
        fprintf(stderr, "%.1fs audio, %.1fms (mel=%.1f enc=%.1f dec=%.1f), %.0fx RTFx\n",
                audio_dur, elapsed * 1000,
                pipeline.last_mel_ms, pipeline.last_enc_ms, pipeline.last_dec_ms,
                audio_dur / elapsed);
    }

    return 0;
}
