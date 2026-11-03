#pragma once

/**
 * @file filters.hpp
 * @brief Image filtering and convolution operations.
 *
 * This module provides various image filters including convolution,
 * median filtering, Gaussian blur, and edge detection (Sobel).
 * The library currently explicitly instantiates these operations for
 * uint8_t images.
 *
 * @see Image for the image container class
 * @see point_ops.hpp for point-wise operations
 */

#include "image.hpp"
#include <array>

/**
 * @brief Applies a generic convolution kernel to an image.
 *
 * Convolution is performed by sliding the kernel over the image and
 * computing the weighted sum of pixels in the kernel's neighborhood.
 * Boundary pixels are handled by checking bounds (no padding).
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @param input Source image to convolve
 * @param output Destination image for result (must have matching dimensions,
 * channels, and stride)
 * @param kernel 2D convolution kernel (must be square and have odd dimensions)
 * @throws std::invalid_argument if the kernel is empty, not square, or even,
 * or the image metadata does not match
 *
 * @note The kernel values are NOT normalized by this function.
 * @note Output values are clamped to [0, 255] range.
 *
 * @par Example
 * @code
 * #include "filters.hpp"
 * #include <vector>
 *
 * Image<uint8_t> input(5, 5, 1);
 * Image<uint8_t> output(5, 5, 1);
 * // 3x3 box blur kernel
 * std::vector<std::vector<float>> kernel = {
 *     {1.0f/9, 1.0f/9, 1.0f/9},
 *     {1.0f/9, 1.0f/9, 1.0f/9},
 *     {1.0f/9, 1.0f/9, 1.0f/9}
 * };
 * convolve(input, output, kernel);
 * @endcode
 */
template<typename PixelType>
void convolve(const Image<PixelType>& input, Image<PixelType>& output, const std::vector<std::vector<float>>& kernel);

/**
 * @brief Applies a generic convolution kernel to an image (std::array version).
 *
 * This overload accepts fixed-size square 2D array kernels for better performance.
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @tparam KernelSize The size of the square kernel (must be odd)
 * @param input Source image to convolve
 * @param output Destination image for result (must have matching dimensions,
 * channels, and stride)
 * @param kernel 2D square convolution kernel using std::array
 * @throws std::invalid_argument if KernelSize is even or the image metadata
 * does not match
 *
 * @note The library currently explicitly instantiates this overload for
 * uint8_t images and 3x3 kernels.
 */
template<typename PixelType, size_t KernelSize>
void convolve(const Image<PixelType>& input, Image<PixelType>& output, const std::array<std::array<float, KernelSize>, KernelSize>& kernel);

/**
 * @brief Applies a median filter to reduce noise.
 *
 * The median filter replaces each pixel with the median value of
 * pixels in its neighborhood. It is effective at removing salt-and-pepper
 * noise while preserving edges better than mean filtering.
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @param input Source image to filter
 * @param output Destination image for result (must have matching dimensions,
 * channels, and stride)
 * @param kernelSize Size of the square neighborhood window (must be odd and > 1)
 * @throws std::invalid_argument if kernelSize is even, <= 1, or image metadata
 * does not match
 *
 * @note The kernelSize determines the window size: kernelSize=3 gives 3x3 window
 */
template<typename PixelType>
void medianFilter(const Image<PixelType>& input, Image<PixelType>& output, size_t kernelSize);

/**
 * @brief Applies Gaussian blur to smooth an image.
 *
 * Gaussian blur uses a kernel derived from the Gaussian distribution.
 * It effectively reduces high-frequency noise and detail, producing
 * a smoothing effect. The blur amount is controlled by the sigma parameter.
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @param input Source image to blur
 * @param output Destination image for result (must have matching dimensions,
 * channels, and stride)
 * @param sigma Standard deviation of the Gaussian distribution (controls blur amount, must be > 0)
 * @throws std::invalid_argument if sigma <= 0 or image metadata does not match
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
template<typename PixelType>
void gaussianBlur(const Image<PixelType>& input, Image<PixelType>& output, float sigma);

/**
 * @brief Applies Gaussian blur with a default sigma value.
 *
 * This is a convenience function that applies Gaussian blur using a fixed
 * 3x3 kernel with sigma ≈ 0.85. This provides a standard amount of smoothing
 * without requiring the user to specify a sigma parameter.
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @param input Source image to blur
 * @param output Destination image for result (must have matching dimensions,
 * channels, and stride)
 * @throws std::invalid_argument if image metadata does not match
 *
 * @note Uses a fixed kernel: [0.0625, 0.125, 0.0625; 0.125, 0.25, 0.125; 0.0625, 0.125, 0.0625]
 * @note For more control over blur amount, use the version with sigma parameter
 */
template<typename PixelType>
void gaussianBlur(const Image<PixelType>& input, Image<PixelType>& output);

/**
 * @brief Applies Sobel edge detection filter.
 *
 * The Sobel filter detects edges by computing the gradient magnitude
 * using two 3x3 kernels (Gx for horizontal, Gy for vertical).
 * The gradient magnitude is: sqrt(Gx^2 + Gy^2)
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @param input Source image; each channel is processed independently
 * @param output Destination image for edge result (must have matching
 * dimensions, channels, and stride)
 * @throws std::invalid_argument if input and output metadata don't match
 *
 * @note For conventional edge detection, apply to grayscale images.
 * @note The Sobel kernels used are:
 *       Gx = [-1  0  1]    Gy = [ 1  2  1]
 *           [-2  0  2]          [ 0  0  0]
 *           [-1  0  1]          [-1 -2 -1]
 *
 * @see rgbToGrayscale() in pnm_funcs.hpp to convert color images to grayscale
 */
template<typename PixelType>
void sobelFilter(const Image<PixelType>& input, Image<PixelType>& output);

/**
 * @brief Applies a sharpening filter to enhance image details.
 *
 * Sharpening enhances edges and fine details using a 3x3 cross-shaped
 * convolution kernel that emphasizes the center pixel while subtracting
 * neighboring pixels.
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @param input Source image to sharpen
 * @param output Destination image for sharpened result (must have matching
 * dimensions, channels, and stride)
 * @throws std::invalid_argument if image metadata does not match
 *
 * @note Uses kernel: [0, -1, 0; -1, 5, -1; 0, -1, 0]
 * @note May amplify noise along with edges
 */
template<typename PixelType>
void sharpeningFilter(const Image<PixelType>& input, Image<PixelType>& output);

/**
 * @brief Applies Laplacian edge detection filter.
 *
 * The Laplacian filter detects edges by computing the second derivative
 * of the image intensity. It highlights regions of rapid intensity change,
 * which typically correspond to edges. The output contains the clamped
 * response for each input channel.
 *
 * @tparam PixelType The pixel data type (typically uint8_t)
 * @param input Source image; each channel is processed independently
 * @param output Destination image for edge result (must have matching
 * dimensions, channels, and stride)
 * @throws std::invalid_argument if image metadata does not match
 *
 * @note Uses kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
 * @note For conventional edge detection, apply to grayscale images.
 * @note Output values are clamped to [0, 255] range
 *
 * @see rgbToGrayscale() in pnm_funcs.hpp to convert color images to grayscale
 */
template<typename PixelType>
void laplacianFilter(const Image<PixelType>& input, Image<PixelType>& output);
