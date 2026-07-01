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
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename PixelType>
    void erode(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize);

    /**
     * @brief Applies grayscale dilation with a square structuring element.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename PixelType>
    void dilate(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize);

    /**
     * @brief Applies morphological opening (erosion followed by dilation).
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename PixelType>
    void open(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize);

    /**
     * @brief Applies morphological closing (dilation followed by erosion).
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename PixelType>
    void close(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize);

    /**
     * @brief Computes the morphological gradient as dilation minus erosion.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename PixelType>
    void morphologicalGradient(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize);

    /**
     * @brief Computes the white top-hat transform as input minus opening.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename PixelType>
    void topHat(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize);

    /**
     * @brief Computes the black top-hat transform as closing minus input.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square structuring element
     */
    template<typename PixelType>
    void bottomHat(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize);
}
