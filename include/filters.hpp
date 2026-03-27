#pragma once

/**
 * @file filters.hpp
 * @brief Image filtering and convolution operations.
 *
 * This module provides various image filters including convolution,
 * median filtering, Gaussian blur, and edge detection (Sobel).
 *
 * @see Image for the image container class
 * @see point_ops.hpp for point-wise operations
 */

#include "image.hpp"

/**
 * @brief Applies a generic convolution kernel to an image.
 *
 * Convolution is performed by sliding the kernel over the image and
 * computing the weighted sum of pixels in the kernel's neighborhood.
 * Boundary pixels are handled by checking bounds (no padding).
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source image to convolve
 * @param output Destination image for result (must have same dimensions as input)
 * @param kernel 2D convolution kernel (must be square and have odd dimensions)
 * @throws std::invalid_argument if kernel is empty, not square, has even size, or dimensions mismatch
 *
 * @note The kernel values are NOT normalized by this function.
 * @note Output values are clamped to [0, 255] range.
 *
 * @par Example
 * @code
 * // 3x3 box blur kernel
 * std::vector<std::vector<float>> kernel = {
 *     {1.0f/9, 1.0f/9, 1.0f/9},
 *     {1.0f/9, 1.0f/9, 1.0f/9},
 *     {1.0f/9, 1.0f/9, 1.0f/9}
 * };
 * convolve(input, output, kernel);
 * @endcode
 */
template<typename T>
void convolve(const Image<T>& input, Image<T>& output, const std::vector<std::vector<float>>& kernel);

/**
 * @brief Applies a median filter to reduce noise.
 *
 * The median filter replaces each pixel with the median value of
 * pixels in its neighborhood. It is effective at removing salt-and-pepper
 * noise while preserving edges better than mean filtering.
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source image to filter
 * @param output Destination image for result (must have same dimensions as input)
 * @param kernelSize Size of the square neighborhood window (must be odd and > 1)
 * @throws std::invalid_argument if kernelSize is even, <= 1, or dimensions mismatch
 *
 * @note The kernelSize determines the window size: kernelSize=3 gives 3x3 window
 */
template<typename T>
void medianFilter(const Image<T>& input, Image<T>& output, size_t kernelSize);

/**
 * @brief Applies Gaussian blur to smooth an image.
 *
 * Gaussian blur uses a kernel derived from the Gaussian distribution.
 * It effectively reduces high-frequency noise and detail, producing
 * a smoothing effect. The blur amount is controlled by the sigma parameter.
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source image to blur
 * @param output Destination image for result (must have same dimensions as input)
 * @param sigma Standard deviation of the Gaussian distribution (controls blur amount, must be > 0)
 * @throws std::invalid_argument if sigma <= 0 or dimensions mismatch
 *
 * @note The kernel size is automatically determined from sigma: size = ceil(sigma * 6) | 1
 * @note Larger sigma values produce stronger blur but require larger kernels
 *
 * @par Example
 * @code
 * gaussianBlur(input, output, 1.0f);  // Light blur
 * gaussianBlur(input, output, 3.0f);  // Strong blur
 * @endcode
 */
template<typename T>
void gaussianBlur(const Image<T>& input, Image<T>& output, float sigma);

/**
 * @brief Applies Sobel edge detection filter.
 *
 * The Sobel filter detects edges by computing the gradient magnitude
 * using two 3x3 kernels (Gx for horizontal, Gy for vertical).
 * The gradient magnitude is: sqrt(Gx^2 + Gy^2)
 *
 * @tparam T The pixel data type (typically uint8_t)
 * @param input Source grayscale image for edge detection
 * @param output Destination image for edge result (must have same dimensions as input)
 * @throws std::invalid_argument if input and output dimensions don't match
 *
 * @note For proper edge detection, apply to grayscale images.
 * @note The Sobel kernels used are:
 *       Gx = [-1  0  1]    Gy = [ 1  2  1]
 *           [-2  0  2]          [ 0  0  0]
 *           [-1  0  1]          [-1 -2 -1]
 *
 * @see rgbToGrayscale() in pnm_funcs.hpp to convert color images to grayscale
 */
template<typename T>
void sobelFilter(const Image<T>& input, Image<T>& output);
