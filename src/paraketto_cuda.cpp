// paraketto_cuda.cpp — Custom CUDA/cuBLAS runtime for Parakeet TDT 0.6B V2
//
// Build: make paraketto.cuda
// Usage: ./paraketto.cuda [--weights FILE] audio.wav

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "conformer.h"

// ---------------------------------------------------------------------------
// Auto-download helpers
// ---------------------------------------------------------------------------

static std::string cache_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/paraketto";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.cache/paraketto";
    return ".";
}

static void ensure_file(const std::string& path, const char* url) {
    if (access(path.c_str(), F_OK) == 0) return;
    std::string dir = path.substr(0, path.rfind('/'));
    if (!dir.empty()) {
        std::string cmd = "mkdir -p '" + dir + "'";
        if (system(cmd.c_str()) != 0) {}  // best-effort
    }
    fprintf(stderr, "Downloading %s...\n", path.c_str());
    std::string cmd = "wget -q --show-progress -O '" + path + "' '" + std::string(url) + "'";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        unlink(path.c_str());
        fprintf(stderr, "Download failed. Get weights from https://huggingface.co/localoptima/paraketto\n");
        exit(1);
    }
}

#include "common.h"
#include "wav.h"
#include "mel.h"
#include "vocab.h"
#include "server.h"

#ifdef EMBEDDED_WEIGHTS
extern "C" {
#  ifdef PARAKETTO_FP8
    extern const uint8_t _binary_paraketto_fp8_bin_start[];
    extern const uint8_t _binary_paraketto_fp8_bin_end[];
#  else
    extern const uint8_t _binary_paraketto_fp16_bin_start[];
    extern const uint8_t _binary_paraketto_fp16_bin_end[];
#  endif
}
#endif

// ---------------------------------------------------------------------------
// Inference pipeline (CUDA backend)
// ---------------------------------------------------------------------------

struct Pipeline {
    Weights weights;
    CudaModel cuda_model;
    MelSpec mel;
    cudaStream_t stream = nullptr;

    // fp8_ready: non-empty = paraketto-fp8.bin exists and is valid (use allocate_only).
    // fp8_target: path to load/save paraketto-fp8.bin (may differ on first run).
    // fp8_prefetch: pre-populated mmap of paraketto-fp8.bin from background thread.
    void init(Weights&& prefetched,
              const std::string& fp8_ready  = "",
              const std::string& fp8_target = "",
              const void* fp8_prefetch = nullptr,
              size_t fp8_prefetch_size = 0) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        weights = std::move(prefetched);

#ifdef PARAKETTO_FP8
        if (!fp8_ready.empty())
            weights.allocate_only();  // FP8 file ready: malloc + assign pointers only
        else
            weights.upload(stream);   // first run: upload FP16 to quantize from
#else
        weights.upload(stream);
#endif

        mel.init();

#ifdef PARAKETTO_FP8
        const char* fp8_path = fp8_target.empty() ? nullptr : fp8_target.c_str();
        cuda_model.init(weights, stream, 16000 * 120 / 160, fp8_path,
                        fp8_prefetch, fp8_prefetch_size);
#else
        cuda_model.init(weights, stream, 16000 * 120 / 160);
#endif
    }

    ~Pipeline() {
        cuda_model.free();
        weights.free();
        if (stream) cudaStreamDestroy(stream);
    }

    double last_mel_ms = 0, last_enc_ms = 0, last_dec_ms = 0;

    std::string transcribe(const float* samples, int num_samples) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // --- 1. Mel spectrogram (fused GPU pipeline) ---
        int n_frames, n_valid;
        mel.compute(samples, num_samples, cuda_model.mel_fp32, n_frames, n_valid, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t_mel = std::chrono::high_resolution_clock::now();

        // --- 2. Encoder (mel_fp32 already on GPU) ---
        int T = cuda_model.encode_gpu(n_valid);
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

        // Host buffer for argmax results (just 2 ints: token, step)
        int argmax_host[2];

        while (t < enc_len) {
            half* joint_out = cuda_model.decode_step(t, last_token);

            // GPU argmax: finds token and step on device, transfers just 2 ints
            dual_argmax_fp16(joint_out, cuda_model.argmax_out,
                              VOCAB_SIZE, D_OUTPUT, stream);
            CUDA_CHECK(cudaMemcpyAsync(argmax_host, cuda_model.argmax_out,
                                        2 * sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int token = argmax_host[0];
            int step = argmax_host[1];

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
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto usage = [&]() {
        fprintf(stderr,
            "Usage: %s [OPTIONS] <wav_file>...\n"
            "\n"
            "Speech-to-text using Paraketto (CUDA/cuBLAS backend).\n"
            "Accepts one or more 16kHz or 24kHz mono WAV files (int16 or float32).\n"
            "Weights are auto-downloaded from HuggingFace on first run.\n"
            "\n"
            "Options:\n"
            "  --weights FILE             Model weights [default: ~/.cache/paraketto/]\n"
            "  --server [[host]:port]     Start HTTP server [default: 0.0.0.0:8080]\n"
            "  -h, --help                 Show this help\n",
            argv[0]);
    };

    if (argc < 2) { usage(); return 1; }

    std::string dir = cache_dir();
    bool custom_weights = false;
#ifdef PARAKETTO_FP8
    std::string weights_path = dir + "/paraketto-fp8.bin";
    std::string fp16_path    = dir + "/paraketto-fp16.bin";
#else
    std::string weights_path = dir + "/paraketto-fp16.bin";
#endif
    std::vector<std::string> wav_files;
    bool server_mode = false;
    std::string server_host = "0.0.0.0";
    int server_port = 8080;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
            custom_weights = true;
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

#ifndef EMBEDDED_WEIGHTS
    if (!custom_weights) {
#  ifdef PARAKETTO_FP8
        ensure_file(weights_path, "https://huggingface.co/localoptima/paraketto/resolve/main/paraketto-fp8.bin");
        // FP16 weights needed only for first-run quantization (if fp8 file is missing/invalid)
#  else
        ensure_file(weights_path, "https://huggingface.co/localoptima/paraketto/resolve/main/paraketto-fp16.bin");
#  endif
    }
#endif

    Weights prefetched;
    std::thread prefetch_thread;

    // For FP8: check whether the fp8 weights file is ready before starting prefetch.
    // Check if paraketto-fp8.bin is present and valid.
    // If yes: allocate_only() + fp8_load (skip paraketto-fp16.bin upload).
    // If no:  upload paraketto-fp16.bin FP16, quantize, save paraketto-fp8.bin.
    std::string fp8_path_for_init;
    void* fp8_prefetch_ptr = nullptr;
    size_t fp8_prefetch_size = 0;
#if defined(PARAKETTO_FP8) && !defined(EMBEDDED_WEIGHTS)
    {
        int fd = open(weights_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            char hdr[12];
            if (read(fd, hdr, 12) == 12
                    && memcmp(hdr, "PRKTFP8", 7) == 0) {
                uint32_t version; memcpy(&version, hdr + 8, 4);
                if (version == FP8_WEIGHTS_VERSION)
                    fp8_path_for_init = weights_path;
            }
            close(fd);
        }
        // FP8 file missing or invalid — need FP16 weights for quantization
        if (fp8_path_for_init.empty() && !custom_weights) {
            ensure_file(fp16_path, "https://huggingface.co/localoptima/paraketto/resolve/main/paraketto-fp16.bin");
        }
    }
#endif

#ifdef EMBEDDED_WEIGHTS
#  ifdef PARAKETTO_FP8
    fp8_path_for_init = "embedded";  // FP8 static: paraketto-fp8.bin is embedded
    prefetched = Weights{};          // allocate_only() computes layout from constants
#  else
    prefetched = Weights::from_embedded(_binary_paraketto_fp16_bin_start,
        _binary_paraketto_fp16_bin_end - _binary_paraketto_fp16_bin_start);
#  endif
#else
    // Prefetch in background while CUDA context inits.
    prefetch_thread = std::thread([&]() {
#  ifdef PARAKETTO_FP8
        if (fp8_path_for_init.empty()) {
            prefetched = Weights::prefetch(fp16_path);  // need FP16 data for quantization
        } else {
            // Pre-populate FP8 file pages while CUDA init runs
            int fd = open(fp8_path_for_init.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st; fstat(fd, &st);
                fp8_prefetch_size = (size_t)st.st_size;
                fp8_prefetch_ptr = mmap(nullptr, st.st_size, PROT_READ,
                                        MAP_PRIVATE | MAP_POPULATE, fd, 0);
                close(fd);
                if (fp8_prefetch_ptr == MAP_FAILED) {
                    fp8_prefetch_ptr = nullptr;
                    fp8_prefetch_size = 0;
                } else {
                    madvise(fp8_prefetch_ptr, st.st_size, MADV_SEQUENTIAL);
                }
            }
        }
#  else
        prefetched = Weights::prefetch(weights_path);
#  endif
    });
#endif

    // Force CUDA driver/context initialization (overlaps with prefetch)
    cudaFree(0);
    auto t_cuda_init = clk::now();

    if (!server_mode) {
        int wav_fd = open(wav_files[0].c_str(), O_RDONLY);
        if (wav_fd >= 0) { posix_fadvise(wav_fd, 0, 0, POSIX_FADV_WILLNEED); close(wav_fd); }
    }

    if (prefetch_thread.joinable()) prefetch_thread.join();
    auto t_prefetch = clk::now();

    Pipeline pipeline;
#ifdef PARAKETTO_FP8
    pipeline.init(std::move(prefetched), fp8_path_for_init, weights_path,
                  fp8_prefetch_ptr, fp8_prefetch_size);
#else
    pipeline.init(std::move(prefetched));
#endif
    auto t_init_done = clk::now();

    // Clean up FP8 prefetch mapping (data is on GPU now)
    if (fp8_prefetch_ptr) {
        munmap(fp8_prefetch_ptr, fp8_prefetch_size);
        fp8_prefetch_ptr = nullptr;
    }

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
        fprintf(stderr, "startup:   %.0f ms (cuda=%.0f prefetch=%.0f load=%.0f warmup=%.0f)\n",
                ms(t_main_start, t_warmup_done), ms(t_main_start, t_cuda_init),
                ms(t_main_start, t_prefetch),
                ms(t_prefetch, t_init_done), ms(t_init_done, t_warmup_done));
        fprintf(stderr, "endpoints: GET /health | POST /transcribe\n");
        fprintf(stderr, "\n");

        run_server(pipeline, server_host, server_port);
        return 0;
    }

    auto t_ready = clk::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    fprintf(stderr, "startup: %.0fms (cuda=%.0f prefetch=%.0f load=%.0f)\n",
            ms(t_main_start, t_ready), ms(t_main_start, t_cuda_init),
            ms(t_main_start, t_prefetch),
            ms(t_prefetch, t_init_done));

    // Process all files
    for (size_t fi = 0; fi < wav_files.size(); fi++) {
        WavData wav = read_wav(wav_files[fi]);
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
