#pragma once

/**
 * @file utils.hpp
 * @brief Utility functions for image processing operations.
 */

#include <cstdint>
#include <vector>
#include <stdexcept>
#include "image.hpp"

/**
 * @brief Clamps a value to a specified range.
 *
 * @tparam T The data type for the value (must support comparison operators)
 * @param value The value to clamp
 * @param min The minimum bound (inclusive)
 * @param max The maximum bound (inclusive)
 * @return The clamped value within [min, max]
 */
template<typename T>
T clamp(T value, T min, T max);

/**
 * @brief Validates that a kernel size is valid for image processing operations.
 *
 * Checks that the kernel size is non-zero and odd (valid for symmetric kernels).
 * This validation is applicable to morphological operations, filtering, and other
 * image processing techniques that require square, symmetric kernels.
 *
 * @param kernel_size The size of the kernel to validate
 * @throws std::invalid_argument If kernel_size is zero or even
 */
void validateKernelSize(size_t kernel_size);

/**
 * @brief Validates that a kernel matrix is valid for image processing operations.
 *
 * Checks that the kernel matrix is:
 * - Not empty
 * - Square (number of rows equals number of columns in each row)
 * - Has odd dimensions (valid for symmetric kernels)
 *
 * @tparam T The data type of the kernel elements
 * @param kernel A 2D vector representing the kernel matrix
 * @throws std::invalid_argument If kernel is empty, not square, or has even dimensions
 */
template<typename T>
void validateKernel(const std::vector<std::vector<T>>& kernel);

/**
 * @brief Checks if two images have identical dimensions.
 *
 * Compares two images to determine if they have the same dimensions
 * (rows, columns, channels). The pixel data is not compared.
 *
 * @tparam T The data type of the image pixels
 * @param img1 First image to compare
 * @param img2 Second image to compare
 * @return true if both images have the same dimensions, false otherwise
 */
template<typename T>
bool imagesEqual(const Image<T>& img1, const Image<T>& img2);
