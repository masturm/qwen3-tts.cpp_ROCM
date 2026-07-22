#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qwen3_tts {

bool wav_encode(const std::vector<float> & samples, int sample_rate,
                std::vector<uint8_t> & out);

bool wav_decode(const uint8_t * data, size_t size,
                std::vector<float> & samples, int & sample_rate);

} // namespace qwen3_tts
