/**
 * @file gpu_utils.cu
 * @brief Implementation of utility functions for CUDA image processing.
 *
 * This file contains implementations of CUDA-side helper and validation
 * functions used throughout the GPU-accelerated image processing operations.
 * These include kernel validation, GpuImage compatibility checks, and
 * device-side utility functions.
 *
 * @see gpu_utils.cuh for the function declarations and detailed documentation
 */

#include "cuda/gpu_utils.cuh"
#include <algorithm>

namespace cuda {
    template<typename T>
    void checkNonEmpty(const std::vector<std::vector<T>>& kernel) {
        if (kernel.empty()) {
            throw std::invalid_argument("Kernel cannot be empty");
        }
    }

    void checkOddSize(size_t size) {
        if (size % 2 == 0) {
            throw std::invalid_argument("Size must be odd");
        }
    }

    template<typename T>
    void checkSquareAndOdd(const std::vector<std::vector<T>>& kernel) {
        checkNonEmpty(kernel);

        const size_t kernelSize = kernel.size();
        for (const auto& row : kernel) {
            if (row.size() != kernelSize) {
                throw std::invalid_argument("Kernel must be square");
            }
        }

        checkOddSize(kernelSize);
    }

    template<typename T, size_t N>
    void checkSquareAndOdd(const std::array<std::array<T, N>, N>& kernel) {
        (void)kernel;
        checkOddSize(N);
    }

    template<typename T>
    void checkSameTypeImages(const GpuImage<T>& first, const GpuImage<T>& second) {
        if (first.rows() != second.rows() ||
            first.cols() != second.cols() ||
            first.channels() != second.channels() ||
            first.stride() != second.stride()) {
            throw std::invalid_argument("Images must have matching dimensions, channels, and stride");
        }
    }

    template<typename T>
    void checkSameTypeImages(const GpuImage<T>& gpuImage, const Image<T>& cpuImage) {
        if (gpuImage.rows() != cpuImage.rows() ||
            gpuImage.cols() != cpuImage.cols() ||
            gpuImage.channels() != cpuImage.channels() ||
            gpuImage.stride() != cpuImage.stride()) {
            throw std::invalid_argument("Images must have matching dimensions, channels, and stride");
        }
    }

    template<typename T>
    void checkSameTypeImages(const Image<T>& cpuImage, const GpuImage<T>& gpuImage) {
        checkSameTypeImages(gpuImage, cpuImage);
    }

    template<typename T>
    void checkSingleChannelImage(const GpuImage<T>& image) {
        if (image.channels() != 1) {
            throw std::invalid_argument("CUDA image operation requires a single-channel image");
        }
    }

    template void checkNonEmpty(const std::vector<std::vector<float>>& kernel);
    template void checkNonEmpty(const std::vector<std::vector<uint8_t>>& kernel);

    template void checkSquareAndOdd(const std::vector<std::vector<float>>& kernel);
    template void checkSquareAndOdd(const std::vector<std::vector<uint8_t>>& kernel);
    template void checkSquareAndOdd(const std::array<std::array<float, 3>, 3>& kernel);

    template void checkSameTypeImages(const GpuImage<uint8_t>& first, const GpuImage<uint8_t>& second);
    template void checkSameTypeImages(const GpuImage<uint8_t>& gpuImage, const Image<uint8_t>& cpuImage);
    template void checkSameTypeImages(const Image<uint8_t>& cpuImage, const GpuImage<uint8_t>& gpuImage);
    template void checkSingleChannelImage(const GpuImage<uint8_t>& image);
} // namespace cuda
