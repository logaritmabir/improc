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
     * @tparam T The data type for the value
     * @param value The value to clamp
     * @param min The minimum bound (inclusive)
     * @param max The maximum bound (inclusive)
     * @return The clamped value within [min, max]
     */
    template<typename T>
    __device__ __forceinline__ T clamp(T value, T min, T max) {
        return value < min ? min : (value > max ? max : value);
    }

    /**
     * @brief Requires that the provided dynamic kernel is not empty.
     *
     * @tparam T The data type of the kernel elements
     * @param kernel A 2D vector kernel to validate
     * @throws std::invalid_argument If @p kernel is empty
     */
    template<typename T>
    void requireNonEmpty(const std::vector<std::vector<T>>& kernel);

    /**
     * @brief Requires that the provided size value is odd.
     *
     * @param size The size value to validate
     * @throws std::invalid_argument If @p size is even
     */
    void requireOddSize(size_t size);

    /**
     * @brief Requires that the provided dynamic kernel is square and has odd width.
     *
     * @tparam T The data type of the kernel elements
     * @param kernel A 2D vector kernel to validate
     * @throws std::invalid_argument If @p kernel is empty, not square, or has even width
     */
    template<typename T>
    void requireSquareAndOdd(const std::vector<std::vector<T>>& kernel);

    /**
     * @brief Requires that the provided fixed-size kernel has odd width.
     *
     * @tparam T The data type of the kernel elements
     * @tparam N The kernel width and height
     * @param kernel A fixed-size square kernel to validate
     * @throws std::invalid_argument If @p N is even
     */
    template<typename T, size_t N>
    void requireSquareAndOdd(const std::array<std::array<T, N>, N>& kernel);

    /**
     * @brief Requires that two GPU images have matching metadata.
     *
     * @tparam T The data type of the image pixels
     * @param first First image to validate
     * @param second Second image to validate
     * @throws std::invalid_argument If the image metadata does not match
     */
    template<typename T>
    void requireSameTypeImages(const GpuImage<T>& first, const GpuImage<T>& second);

    /**
     * @brief Requires that a GPU image and CPU image have matching metadata.
     *
     * @tparam T The data type of the image pixels
     * @param gpuImage GPU image to validate
     * @param cpuImage CPU image to validate
     * @throws std::invalid_argument If the image metadata does not match
     */
    template<typename T>
    void requireSameTypeImages(const GpuImage<T>& gpuImage, const Image<T>& cpuImage);

    /**
     * @brief Requires that a CPU image and GPU image have matching metadata.
     *
     * @tparam T The data type of the image pixels
     * @param cpuImage CPU image to validate
     * @param gpuImage GPU image to validate
     * @throws std::invalid_argument If the image metadata does not match
     */
    template<typename T>
    void requireSameTypeImages(const Image<T>& cpuImage, const GpuImage<T>& gpuImage);

    /**
     * @brief Requires that a GPU image contains exactly one channel.
     *
     * @tparam T The data type of the image pixels
     * @param image Image to validate
     * @throws std::invalid_argument If @p image is not single-channel
     */
    template<typename T>
    void requireSingleChannelImage(const GpuImage<T>& image);

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
