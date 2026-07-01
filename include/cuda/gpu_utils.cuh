#pragma once

/**
 * @file gpu_utils.cuh
 * @brief Utility functions for CUDA image processing operations.
 *
 * This module mirrors the CPU-side utility helpers for CUDA-facing code. It
 * provides kernel validation helpers, GpuImage compatibility checks, a generic
 * clamp helper, and lightweight device indexing utilities.
 *
 * @note This is a CUDA header file and requires NVCC or a compatible compiler.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
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
    __device__ __forceinline__ ValueType clamp(ValueType value, ValueType min, ValueType max) {
        return value < min ? min : (value > max ? max : value);
    }

    /**
     * @brief Checks that the provided dynamic kernel is not empty.
     *
     * @tparam ElementType The data type of the kernel elements
     * @param kernel A 2D vector kernel to validate
     * @throws std::invalid_argument If @p kernel is empty
     */
    template<typename ElementType>
    void checkNonEmpty(const std::vector<std::vector<ElementType>>& kernel);

    /**
     * @brief Checks that the provided size value is odd.
     *
     * @param size The size value to validate
     * @throws std::invalid_argument If @p size is even
     */
    void checkOddSize(size_t size);

    /**
     * @brief Checks that the provided dynamic kernel is square and has odd width.
     *
     * @tparam ElementType The data type of the kernel elements
     * @param kernel A 2D vector kernel to validate
     * @throws std::invalid_argument If @p kernel is empty, not square, or has even width
     */
    template<typename ElementType>
    void checkSquareAndOdd(const std::vector<std::vector<ElementType>>& kernel);

    /**
     * @brief Checks that the provided fixed-size kernel has odd width.
     *
     * @tparam ElementType The data type of the kernel elements
     * @tparam KernelSize The kernel width and height
     * @param kernel A fixed-size square kernel to validate
     * @throws std::invalid_argument If @p KernelSize is even
     */
    template<typename ElementType, size_t KernelSize>
    void checkSquareAndOdd(const std::array<std::array<ElementType, KernelSize>, KernelSize>& kernel);

    /**
     * @brief Checks that two GPU images have matching metadata.
     *
     * @tparam PixelType The data type of the image pixels
     * @param first First image to validate
     * @param second Second image to validate
     * @throws std::invalid_argument If the image metadata does not match
     */
    template<typename PixelType>
    void checkSameTypeImages(const GpuImage<PixelType>& first, const GpuImage<PixelType>& second);

    /**
     * @brief Checks that a GPU image and CPU image have matching metadata.
     *
     * @tparam PixelType The data type of the image pixels
     * @param gpuImage GPU image to validate
     * @param cpuImage CPU image to validate
     * @throws std::invalid_argument If the image metadata does not match
     */
    template<typename PixelType>
    void checkSameTypeImages(const GpuImage<PixelType>& gpuImage, const Image<PixelType>& cpuImage);

    /**
     * @brief Checks that a CPU image and GPU image have matching metadata.
     *
     * @tparam PixelType The data type of the image pixels
     * @param cpuImage CPU image to validate
     * @param gpuImage GPU image to validate
     * @throws std::invalid_argument If the image metadata does not match
     */
    template<typename PixelType>
    void checkSameTypeImages(const Image<PixelType>& cpuImage, const GpuImage<PixelType>& gpuImage);

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
     * @param x Row index
     * @param y Column index
     * @return Linear index (x * cols + y)
     */
    __device__ __forceinline__ size_t index(size_t cols, size_t x, size_t y) {
        return x * cols + y;
    }

} // namespace cuda
