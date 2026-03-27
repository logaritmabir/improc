#pragma once

/**
 * @file kernels.cuh
 * @brief CUDA convolution kernels for GPU-accelerated image processing.
 *
 * This module provides CUDA kernels for performing image convolution
 * on the GPU, significantly accelerating convolution operations compared
 * to CPU-based implementations.
 *
 * @note This is a CUDA header file and requires NVCC or a compatible compiler.
 * @see GpuImage for the GPU image container
 * @see convolve() in filters.hpp for the CPU-side convolution
 */

#include "gpu_image.cuh"
#include <cstddef>

namespace cuda {
    /**
     * @brief Computes a linear index for 2D data.
     *
     * @param cols Number of columns
     * @param x Row index
     * @param y Column index
     * @return Linear index (x * cols + y)
     */
    __device__ __forceinline__ size_t index(size_t cols, size_t x, size_t y);

    /**
     * @brief CUDA kernel for image convolution.
     *
     * This kernel is launched with a 2D grid of threads, where each thread
     * computes the convolution for one output pixel. The kernel slides the
     * convolution kernel over the input image and computes weighted sums.
     *
     * @tparam T Pixel data type
     * @param input Input image data on device
     * @param output Output image data on device
     * @param kernel Convolution kernel data on device
     * @param rows Number of rows in the image
     * @param cols Number of columns in the image
     * @param kernelWidth Width of the square kernel
     */
    template<typename T>
    __global__ void convolution(const T* input, T* output, const float* kernel, size_t rows, size_t cols, size_t kernelWidth);

    /**
     * @brief Performs GPU-accelerated convolution on GpuImage objects.
     *
     * This is the high-level function for GPU convolution. It handles
     * kernel allocation on the device, kernel execution, and cleanup.
     *
     * @tparam T Pixel data type
     * @param d_input Input image on GPU (must have same dimensions as output)
     * @param d_output Output image on GPU (must have same dimensions as input)
     * @param h_kernel Convolution kernel on host (must be 1D array, square kernel)
     * @param h_kernelSize Size of the square kernel (must be odd)
     * @throws std::runtime_error if dimensions don't match or kernel size is even
     *
     * @note Uses a 16x16 thread block configuration
     * @note Output values are NOT clamped (may exceed 255 for uint8_t)
     */
    template<typename T>
    void convolve(const GpuImage<T>& d_input, GpuImage<T>& d_output, const float* h_kernel, size_t h_kernelSize);
}
