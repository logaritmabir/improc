#include "cuda/gpu_utils.cuh"
#include <algorithm>

namespace cuda {
    template<typename T>
    void requireNonEmpty(const std::vector<std::vector<T>>& kernel) {
        if (kernel.empty()) {
            throw std::invalid_argument("Kernel cannot be empty");
        }
    }

    void requireOddSize(size_t size) {
        if (size % 2 == 0) {
            throw std::invalid_argument("Size must be odd");
        }
    }

    template<typename T>
    void requireSquareAndOdd(const std::vector<std::vector<T>>& kernel) {
        requireNonEmpty(kernel);

        const size_t kernelSize = kernel.size();
        for (const auto& row : kernel) {
            if (row.size() != kernelSize) {
                throw std::invalid_argument("Kernel must be square");
            }
        }

        requireOddSize(kernelSize);
    }

    template<typename T, size_t N>
    void requireSquareAndOdd(const std::array<std::array<T, N>, N>& kernel) {
        (void)kernel;
        requireOddSize(N);
    }

    template<typename T>
    void requireSameTypeImages(const GpuImage<T>& first, const GpuImage<T>& second) {
        if (first.rows() != second.rows() ||
            first.cols() != second.cols() ||
            first.channels() != second.channels() ||
            first.stride() != second.stride()) {
            throw std::invalid_argument("Images must have matching dimensions, channels, and stride");
        }
    }

    template<typename T>
    void requireSameTypeImages(const GpuImage<T>& gpuImage, const Image<T>& cpuImage) {
        if (gpuImage.rows() != cpuImage.rows() ||
            gpuImage.cols() != cpuImage.cols() ||
            gpuImage.channels() != cpuImage.channels() ||
            gpuImage.stride() != cpuImage.stride()) {
            throw std::invalid_argument("Images must have matching dimensions, channels, and stride");
        }
    }

    template<typename T>
    void requireSameTypeImages(const Image<T>& cpuImage, const GpuImage<T>& gpuImage) {
        requireSameTypeImages(gpuImage, cpuImage);
    }

    template<typename T>
    void requireSingleChannelImage(const GpuImage<T>& image) {
        if (image.channels() != 1) {
            throw std::invalid_argument("CUDA image operation requires a single-channel image");
        }
    }

    template void requireNonEmpty(const std::vector<std::vector<float>>& kernel);
    template void requireNonEmpty(const std::vector<std::vector<uint8_t>>& kernel);

    template void requireSquareAndOdd(const std::vector<std::vector<float>>& kernel);
    template void requireSquareAndOdd(const std::vector<std::vector<uint8_t>>& kernel);
    template void requireSquareAndOdd(const std::array<std::array<float, 3>, 3>& kernel);

    template void requireSameTypeImages(const GpuImage<uint8_t>& first, const GpuImage<uint8_t>& second);
    template void requireSameTypeImages(const GpuImage<uint8_t>& gpuImage, const Image<uint8_t>& cpuImage);
    template void requireSameTypeImages(const Image<uint8_t>& cpuImage, const GpuImage<uint8_t>& gpuImage);
    template void requireSingleChannelImage(const GpuImage<uint8_t>& image);
} // namespace cuda
