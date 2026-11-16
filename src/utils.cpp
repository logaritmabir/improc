/**
 * @file utils.cpp
 * @brief Implementation of utility functions for image processing operations.
 *
 * This file contains the non-template kernel-size validation function.
 *
 * @see utils.hpp for the function declarations and detailed documentation
 */

#include "utils.hpp"

void requireOddKernelSize(size_t kernelSize) {
    if(kernelSize % 2 == 0)
        throw std::invalid_argument("Kernel size must be odd");
}
