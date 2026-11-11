/**
 * @file gpu_utils.cu
 * @brief Implementation of utility functions for CUDA image processing.
 *
 * This file contains implementations of host-side validation helpers used
 * throughout GPU-accelerated image processing operations. Device-side
 * clamp and indexing helpers are defined inline in gpu_utils.cuh.
 *
 * @see gpu_utils.cuh for the function declarations and detailed documentation
 */

#include "cuda/gpu_utils.cuh"
#include <stdexcept>

namespace cuda {
    void requireOddKernelSize(size_t kernelSize) {
        if (kernelSize % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd");
        }
    }

    template<typename FirstImageType, typename SecondImageType>
    void checkMatchingImageMetadata(const FirstImageType& first, const SecondImageType& second) {
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

    template void checkMatchingImageMetadata(const GpuImage<uint8_t>& first, const GpuImage<uint8_t>& second);
    template void checkMatchingImageMetadata(const GpuImage<uint8_t>& gpuImage, const Image<uint8_t>& cpuImage);

    template void checkSingleChannelImage(const GpuImage<uint8_t>& image);
} // namespace cuda
