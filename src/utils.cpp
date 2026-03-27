#include "utils.hpp"

template<typename T>
T clamp(T value, T min, T max) {
    return std::min(std::max(value, min), max);
}

template float clamp(float value, float min, float max);
template uint8_t clamp(uint8_t value, uint8_t min, uint8_t max);
