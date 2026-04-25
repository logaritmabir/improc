#pragma once

/**
 * @file morph.cuh
 * @brief CUDA morphological image operations.
 *
 * Declares erosion, dilation, and higher-level morphology routines for
 * single-channel @ref cuda::GpuImage data.
 */

#include "cuda/gpu_image.cuh"
#include <cstddef>

namespace cuda {
    /**
     * @brief Applies grayscale erosion with a square structuring element.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename T>
    void erode(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);

    /**
     * @brief Applies grayscale dilation with a square structuring element.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename T>
    void dilate(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);

    /**
     * @brief Applies morphological opening (erosion followed by dilation).
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename T>
    void open(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);

    /**
     * @brief Applies morphological closing (dilation followed by erosion).
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename T>
    void close(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);

    /**
     * @brief Computes the morphological gradient as dilation minus erosion.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename T>
    void morphologicalGradient(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);

    /**
     * @brief Computes the white top-hat transform as input minus opening.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename T>
    void topHat(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);

    /**
     * @brief Computes the black top-hat transform as closing minus input.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename T>
    void bottomHat(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);
}
