#include "utils.hpp"
#include <algorithm>

template<typename T>
T clamp(T value, T min, T max) {
    return std::min(std::max(value, min), max);
}

void requireOddSize(size_t size) {
    if(size % 2 == 0)
        throw std::invalid_argument("Size must be odd");
}

template<typename T>
void requireSquareAndOdd(const std::vector<std::vector<T>>& kernel) {
    if(kernel.empty())
        throw std::invalid_argument("Kernel cannot be empty");

    const size_t kernelSize = kernel.size();
    for(const auto& row : kernel) {
        if(row.size() != kernelSize)
            throw std::invalid_argument("Kernel must be square");
    }

    requireOddSize(kernelSize);
}

template<typename T, size_t N>
void requireSquareAndOdd(const std::array<std::array<T, N>, N>& kernel) {
    (void)kernel;
    requireOddSize(N);
}

template<typename T>
void requireSameTypeImages(const Image<T>& first, const Image<T>& second) {
    if(first.rows() != second.rows() ||
       first.cols() != second.cols() ||
       first.channels() != second.channels() ||
       first.stride() != second.stride()) {
        throw std::invalid_argument("Images must have matching dimensions, channels, and stride");
    }
}

template float clamp(float value, float min, float max);
template int clamp(int value, int min, int max);
template uint8_t clamp(uint8_t value, uint8_t min, uint8_t max);
template void requireSquareAndOdd(const std::vector<std::vector<float>>& kernel);
template void requireSquareAndOdd(const std::vector<std::vector<uint8_t>>& kernel);
template void requireSquareAndOdd(const std::array<std::array<float, 3>, 3>& kernel);
template void requireSameTypeImages(const Image<uint8_t>& first, const Image<uint8_t>& second);
template void requireSameTypeImages(const Image<int>& first, const Image<int>& second);
