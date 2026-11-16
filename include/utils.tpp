#pragma once

#include <algorithm>
#include <stdexcept>

template<typename ValueType>
ValueType clamp(ValueType value, ValueType min, ValueType max) {
    return std::min(std::max(value, min), max);
}

template<typename ElementType>
void requireNonEmpty(const std::vector<std::vector<ElementType>>& kernel) {
    if (kernel.empty()) {
        throw std::invalid_argument("Kernel cannot be empty");
    }
}

template<typename ElementType>
void requireSquareAndOdd(const std::vector<std::vector<ElementType>>& kernel) {
    requireNonEmpty(kernel);

    const size_t kernelSize = kernel.size();
    for (const auto& row : kernel) {
        if (row.size() != kernelSize) {
            throw std::invalid_argument("Kernel must be square");
        }
    }

    requireOddKernelSize(kernelSize);
}

template<typename ElementType, size_t KernelSize>
void requireSquareAndOdd(const std::array<std::array<ElementType, KernelSize>, KernelSize>& kernel) {
    (void)kernel;
    requireOddKernelSize(KernelSize);
}

template<typename PixelType>
void requireSameTypeImages(const Image<PixelType>& first, const Image<PixelType>& second) {
    if (first.rows() != second.rows() ||
        first.cols() != second.cols() ||
        first.channels() != second.channels() ||
        first.stride() != second.stride()) {
        throw std::invalid_argument("Images must have matching dimensions, channels, and stride");
    }
}
