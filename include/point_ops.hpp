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
 * @brief Applies binary thresholding to a grayscale image.
 *
 * Each pixel is set to 255 if its value is >= threshold, otherwise set to 0.
 * This creates a binary (black and white) image from a grayscale source.
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source grayscale image (must have exactly 1 channel)
 * @param output Destination binary image (must have same dimensions and exactly 1 channel as input)
 * @param thresholdValue The threshold value (typically 0-255 for uint8_t)
 * @throws std::invalid_argument if images are not single-channel or dimensions don't match
 */
template<typename T>
void threshold(const Image<T>& input, Image<T>& output, T thresholdValue);

/**
 * @brief Applies histogram equalization to a grayscale image.
 *
 * Histogram equalization redistributes pixel intensity values across the full
 * range to improve contrast. This operation maps the cumulative distribution
 * function (CDF) of the input histogram to a uniform distribution, effectively
 * spreading out pixel values more evenly throughout the intensity range.
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source grayscale image to equalize
 * @param output Destination image where the equalized result is stored (must have same dimensions as input)
 * @throws std::invalid_argument if input and output dimensions don't match
 *
 * @note Works best on grayscale images. For color images, consider applying to each channel separately.
 * @note The output range is normalized to [0, 255] for proper display.
 */
template<typename T>
void histogramEqualization(const Image<T>& input, Image<T>& output);
