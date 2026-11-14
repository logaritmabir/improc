#pragma once

#include <stdexcept>

namespace cuda {
    template<typename ValueType>
    __device__ __forceinline__ ValueType clamp(ValueType value, ValueType min, ValueType max) {
        return value < min ? min : (value > max ? max : value);
    }

    template<typename FirstImageType, typename SecondImageType>
    void checkMatchingImageMetadata(
        const FirstImageType& first,
        const SecondImageType& second) {
        if (first.rows() != second.rows() ||
            first.cols() != second.cols() ||
            first.channels() != second.channels() ||
            first.stride() != second.stride()) {
            throw std::invalid_argument("Images must have matching dimensions, channels, and stride");
        }
    }

    template<typename PixelType>
    void checkSingleChannelImage(const GpuImage<PixelType>& image) {
        if (image.channels() != 1) {
            throw std::invalid_argument("CUDA image operation requires a single-channel image");
        }
    }

    __device__ __forceinline__ size_t linearIndex2D(size_t cols, size_t row, size_t col) {
        return row * cols + col;
    }
}
