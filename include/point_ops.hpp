#pragma once

/**
 * @file point_ops.hpp
 * @brief Point-wise image transformation operations.
 *
 * This module provides functions for applying point-wise (per-pixel)
 * operations to images without requiring neighborhood information.
 *
 * @see Image for the image container class
 */

#include "image.hpp"

/**
 * @brief Inverts the pixel values of an image.
 *
 * For each pixel, the output value is calculated as: output = 255 - input
 * This operation inverts the image colors (photographic negative effect).
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source image to invert
 * @param output Destination image for inverted result (must have same dimensions as input)
 * @throws std::invalid_argument if input and output dimensions don't match
 */
template<typename T>
void invert(const Image<T>& input, Image<T>& output);

/**
 * @brief Applies binary thresholding to an image.
 *
 * Each pixel is set to 255 if its value is >= threshold, otherwise set to 0.
 * This creates a binary (black and white) image from a grayscale source.
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source grayscale image
 * @param output Destination binary image (must have same dimensions as input)
 * @param thresholdValue The threshold value (typically 0-255 for uint8_t)
 * @throws std::invalid_argument if input and output dimensions don't match
 */
template<typename T>
void threshold(const Image<T>& input, Image<T>& output, T thresholdValue);
