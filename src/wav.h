// wav.h — WAV file reader (16kHz mono, int16 or float32, with 24kHz→16kHz resampling)
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Anti-aliasing FIR low-pass filter + linear interpolation resampler (24kHz → 16kHz).
// 31-tap Hamming-windowed sinc, cutoff 7.5kHz @ 24kHz (-6dB at cutoff, -75dB @ 10kHz).
static inline void resample_24k_to_16k(std::vector<float>& samples) {
    static constexpr int NTAPS = 31;
    static constexpr int HALF = NTAPS / 2;
    static constexpr float FIR[NTAPS] = {
        -1.5701712028e-03f,  1.4493850030e-03f,  1.1235023636e-03f, -4.4573699503e-03f,
         2.5741981710e-03f,  6.9852126238e-03f, -1.3014500834e-02f,  1.1998184958e-17f,
         2.4733691186e-02f, -2.5617997556e-02f, -1.8779901739e-02f,  6.7540830344e-02f,
        -3.7078022474e-02f, -1.0818414642e-01f,  2.9144768982e-01f,  6.2569520132e-01f,
         2.9144768982e-01f, -1.0818414642e-01f, -3.7078022474e-02f,  6.7540830344e-02f,
        -1.8779901739e-02f, -2.5617997556e-02f,  2.4733691186e-02f,  1.1998184958e-17f,
        -1.3014500834e-02f,  6.9852126238e-03f,  2.5741981710e-03f, -4.4573699503e-03f,
         1.1235023636e-03f,  1.4493850030e-03f, -1.5701712028e-03f,
    };

    const int n_in = (int)samples.size();
    if (n_in < NTAPS) return;

    // Step 1: Apply FIR low-pass filter (anti-alias before decimation)
    std::vector<float> filtered(n_in);
    for (int i = 0; i < n_in; i++) {
        float sum = 0;
        for (int j = 0; j < NTAPS; j++) {
            int idx = i - HALF + j;
            if (idx >= 0 && idx < n_in) sum += samples[idx] * FIR[j];
        }
        filtered[i] = sum;
    }

    // Step 2: Resample 2/3 ratio via linear interpolation (1.5 input samples per output)
    const int n_out = n_in * 2 / 3;
    samples.resize(n_out);
    for (int i = 0; i < n_out; i++) {
        float pos = i * 1.5f;
        int idx = (int)pos;
        float frac = pos - idx;
        float a = filtered[idx];
        float b = (idx + 1 < n_in) ? filtered[idx + 1] : a;
        samples[i] = a + frac * (b - a);
    }
}

struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
};

static inline WavData read_wav(const std::string& path) {
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
    if (sample_rate != 16000 && sample_rate != 24000) { fprintf(stderr, "Need 16kHz (or 24kHz), got %dHz: %s\n(resample with: ffmpeg -i input.wav -ar 16000 output.wav)\n", sample_rate, path.c_str()); std::exit(1); }

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
    if (wav.sample_rate == 24000) { resample_24k_to_16k(wav.samples); wav.sample_rate = 16000; }
    return wav;
}

static inline WavData read_wav_from_memory(const char* buf, size_t len) {
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

    if (!raw_data || num_channels != 1 || (sample_rate != 16000 && sample_rate != 24000)) return {};

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
    if (wav.sample_rate == 24000) { resample_24k_to_16k(wav.samples); wav.sample_rate = 16000; }
    return wav;
}
