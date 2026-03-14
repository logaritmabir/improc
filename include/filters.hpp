#pragma once
#include "image.hpp"

/**
 * @file filters.hpp
 * @brief Image filtering operations for PNM images.
 *
 * This module provides filtering functions for processing images loaded from PNM format.
 */

/**
 * @brief Applies 2D convolution to an image with a custom kernel.
 *
 * Performs convolution using the provided kernel on each pixel of the input image.
 * Border pixels are handled by clipping to image boundaries.
 *
 * @tparam T Data type for pixel values (typically uint8_t)
 * @param input Input image (typically loaded from PNM format)
 * @param output Output image with convolution applied
 * @param kernel 2D convolution kernel (square matrix)
 * @param divideBy Normalization divisor for kernel sum (ensures proper scaling)
 *
 * @note Input images should be loaded from PNM format files.
 */
template<typename T>
void convolve(const Image<T>& input, Image<T>& output, const std::vector<std::vector<T>>& kernel, T divideBy);
