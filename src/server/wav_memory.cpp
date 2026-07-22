#include "server/wav_memory.h"

#include <cstring>

namespace qwen3_tts {

namespace {

void append_bytes(std::vector<uint8_t> & out, const void * src, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(src);
    out.insert(out.end(), p, p + n);
}

} // namespace

bool wav_encode(const std::vector<float> & samples, int sample_rate,
                std::vector<uint8_t> & out) {
    if (sample_rate <= 0) {
        return false;
    }

    const uint16_t num_channels    = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate       = (uint32_t) sample_rate * num_channels * bits_per_sample / 8;
    const uint16_t block_align     = num_channels * bits_per_sample / 8;
    const uint32_t data_size       = (uint32_t) (samples.size() * block_align);
    const uint32_t file_size       = 36 + data_size;

    out.clear();
    out.reserve(44 + data_size);

    append_bytes(out, "RIFF", 4);
    append_bytes(out, &file_size, 4);
    append_bytes(out, "WAVE", 4);

    append_bytes(out, "fmt ", 4);
    const uint32_t fmt_size     = 16;
    const uint16_t audio_format = 1; // PCM
    const uint32_t sr           = (uint32_t) sample_rate;
    append_bytes(out, &fmt_size, 4);
    append_bytes(out, &audio_format, 2);
    append_bytes(out, &num_channels, 2);
    append_bytes(out, &sr, 4);
    append_bytes(out, &byte_rate, 4);
    append_bytes(out, &block_align, 2);
    append_bytes(out, &bits_per_sample, 2);

    append_bytes(out, "data", 4);
    append_bytes(out, &data_size, 4);

    for (float s : samples) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t pcm = (int16_t) (s * 32767.0f);
        append_bytes(out, &pcm, 2);
    }

    return true;
}

namespace {

struct reader {
    const uint8_t * p;
    const uint8_t * end;

    bool read(void * dst, size_t n) {
        if ((size_t)(end - p) < n) return false;
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }

    bool skip(size_t n) {
        if ((size_t)(end - p) < n) return false;
        p += n;
        return true;
    }
};

} // namespace

bool wav_decode(const uint8_t * data, size_t size,
                std::vector<float> & samples, int & sample_rate) {
    if (!data || size < 44) return false;

    reader r{data, data + size};

    char riff[4];
    if (!r.read(riff, 4) || std::memcmp(riff, "RIFF", 4) != 0) return false;
    uint32_t _file_size;
    if (!r.read(&_file_size, 4)) return false;
    char wave[4];
    if (!r.read(wave, 4) || std::memcmp(wave, "WAVE", 4) != 0) return false;

    uint16_t audio_format    = 0;
    uint16_t num_channels    = 0;
    uint32_t sr              = 0;
    uint16_t bits_per_sample = 0;
    bool     have_fmt        = false;

    while (r.p < r.end) {
        char     chunk_id[4];
        uint32_t chunk_size;
        if (!r.read(chunk_id, 4)) return false;
        if (!r.read(&chunk_size, 4)) return false;

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16) return false;
            if (!r.read(&audio_format, 2)) return false;
            if (!r.read(&num_channels, 2)) return false;
            if (!r.read(&sr, 4)) return false;
            if (!r.skip(6)) return false; // byte rate + block align
            if (!r.read(&bits_per_sample, 2)) return false;
            if (chunk_size > 16) {
                if (!r.skip(chunk_size - 16)) return false;
            }
            have_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (!have_fmt || num_channels == 0) return false;
            sample_rate = (int) sr;

            if (audio_format == 1) {
                if (bits_per_sample == 16) {
                    int n = (int) (chunk_size / (2 * num_channels));
                    samples.resize(n);
                    std::vector<int16_t> raw((size_t) n * num_channels);
                    if (!r.read(raw.data(), (size_t) n * num_channels * 2)) return false;
                    for (int i = 0; i < n; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[(size_t) i * num_channels + c] / 32768.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                    return true;
                } else if (bits_per_sample == 32) {
                    int n = (int) (chunk_size / (4 * num_channels));
                    samples.resize(n);
                    std::vector<int32_t> raw((size_t) n * num_channels);
                    if (!r.read(raw.data(), (size_t) n * num_channels * 4)) return false;
                    for (int i = 0; i < n; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[(size_t) i * num_channels + c] / 2147483648.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                    return true;
                }
                return false;
            } else if (audio_format == 3) {
                int n = (int) (chunk_size / (4 * num_channels));
                samples.resize(n);
                std::vector<float> raw((size_t) n * num_channels);
                if (!r.read(raw.data(), (size_t) n * num_channels * 4)) return false;
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[(size_t) i * num_channels + c];
                    }
                    samples[i] = sum / num_channels;
                }
                return true;
            }
            return false;
        } else {
            if (!r.skip(chunk_size)) return false;
        }
    }

    return false;
}

} // namespace qwen3_tts
