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
} // namespace cuda
