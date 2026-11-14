#pragma once

/**
 * @file gpu_utils.cuh
 * @brief Utility functions for CUDA image processing operations.
 *
 * This module provides GpuImage compatibility checks, a generic clamp helper,
 * and lightweight device indexing utilities.
 *
 * @note This is a CUDA header file and requires NVCC or a compatible compiler.
 */

#include <cstddef>
#include "cuda/gpu_image.cuh"

namespace cuda {
    /**
     * @brief Clamps a value to a specified range.
     *
     * @tparam ValueType The data type for the value
     * @param value The value to clamp
     * @param min The minimum bound (inclusive)
     * @param max The maximum bound (inclusive)
     * @return The clamped value within [min, max]
     */
    template<typename ValueType>
    __device__ __forceinline__ ValueType clamp(ValueType value, ValueType min, ValueType max);

    /**
     * @brief Requires that the provided kernel size is odd.
     *
     * @param kernelSize The kernel size to validate
     * @throws std::invalid_argument If @p kernelSize is even
     */
    void requireOddKernelSize(size_t kernelSize);

    /**
     * @brief Checks that two images have matching metadata.
     *
     * @tparam FirstImageType Type of the first image
     * @tparam SecondImageType Type of the second image
     * @param first First image to validate
     * @param second Second image to validate
     * @throws std::invalid_argument If dimensions, channels, or stride differ
     */
    template<typename FirstImageType, typename SecondImageType>
    void checkMatchingImageMetadata(
        const FirstImageType& first,
        const SecondImageType& second);

    /**
     * @brief Checks that a GPU image contains exactly one channel.
     *
     * @tparam PixelType The data type of the image pixels
     * @param image Image to validate
     * @throws std::invalid_argument If @p image is not single-channel
     */
    template<typename PixelType>
    void checkSingleChannelImage(const GpuImage<PixelType>& image);

    /**
     * @brief Computes a linear index for 2D data.
     *
     * @param cols Number of columns
     * @param row Row index
     * @param col Column index
     * @return Linear index (row * cols + col)
     */
    __device__ __forceinline__ size_t linearIndex2D(size_t cols, size_t row, size_t col);

} // namespace cuda

#include "cuda/gpu_utils.tpp"
