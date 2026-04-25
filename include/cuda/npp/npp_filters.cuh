#pragma once

/**
 * @file npp_filters.cuh
 * @brief NPP-backed CUDA filtering helpers.
 *
 * Declares wrappers around NVIDIA Performance Primitives (NPP) filters that
 * operate on @ref cuda::GpuImage buffers.
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
     * @param h_kernel Host-side square convolution kernel
     * @param h_kernelSize Width of the square kernel
     */
    void convolution(
        const cuda::GpuImage<uint8_t>& d_input, 
        cuda::GpuImage<uint8_t>& d_output, 
        const float* h_kernel, 
        size_t h_kernelSize);
} // namespace npp
