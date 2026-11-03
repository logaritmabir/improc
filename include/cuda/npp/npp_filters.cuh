#pragma once

/**
 * @file npp_filters.cuh
 * @brief NPP-backed CUDA filtering helpers.
 *
 * Declares an NVIDIA Performance Primitives (NPP) convolution wrapper for
 * @ref cuda::GpuImage buffers.
 */

#include "nppi.h"
#include "cuda/gpu_image.cuh"
#include "cuda/gpu_utils.cuh"

namespace npp {
    /**
     * @brief Applies an NPP convolution to a single-channel 8-bit GPU image.
     *
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param h_kernel Host-side square convolution kernel in row-major order
     * @param h_kernelSize Width of the square kernel
     * @pre @p h_kernel points to at least @p h_kernelSize squared float values.
     * Both GPU images must own allocated device buffers.
     * @throws std::invalid_argument If the images are incompatible, the input
     * is not single-channel, or @p h_kernelSize is even
     * @throws std::runtime_error If an NPP or CUDA operation fails
     */
    void convolution(
        const cuda::GpuImage<uint8_t>& d_input, 
        cuda::GpuImage<uint8_t>& d_output, 
        const float* h_kernel, 
        size_t h_kernelSize);
} // namespace npp
