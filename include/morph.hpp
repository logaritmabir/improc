#pragma once

#include "image.hpp"

/**
 * @file morph.hpp
 * @brief Morphological operations for image processing.
 * 
 * This file contains template functions for applying morphological operations
 * (erosion and dilation) to grayscale images.
 */

/**
 * @brief Performs morphological erosion on a grayscale image.
 * 
 * Erosion replaces each pixel with the minimum value from its neighborhood
 * defined by the kernel size. This operation shrinks white regions and expands
 * black regions in the image.
 * 
 * @tparam T The data type of the image (typically uint8_t or float)
 * @param input The source grayscale image (must have exactly 1 channel)
 * @param output The destination grayscale image where the eroded result is stored (must have exactly 1 channel)
 * @param kernel_size The size of the square kernel (must be odd, e.g., 3, 5, 7)
 * 
 * @note Input and output images must have the same dimensions and exactly 1 channel.
 * @note The kernel radius determines the neighborhood: radius = (kernel_size - 1) / 2
 * 
 * @throws std::invalid_argument If kernel_size is even, input/output dimensions don't match, or images are not single-channel
 */
template<typename T>
void erode(const Image<T>& input, Image<T>& output, size_t kernel_size);

/**
 * @brief Performs morphological dilation on a grayscale image.
 * 
 * Dilation replaces each pixel with the maximum value from its neighborhood
 * defined by the kernel size. This operation expands white regions and shrinks
 * black regions in the image.
 * 
 * @tparam T The data type of the image (typically uint8_t or float)
 * @param input The source grayscale image (must have exactly 1 channel)
 * @param output The destination grayscale image where the dilated result is stored (must have exactly 1 channel)
 * @param kernel_size The size of the square kernel (must be odd, e.g., 3, 5, 7)
 * 
 * @note Input and output images must have the same dimensions and exactly 1 channel.
 * @note The kernel radius determines the neighborhood: radius = (kernel_size - 1) / 2
 * 
 * @throws std::invalid_argument If kernel_size is even, input/output dimensions don't match, or images are not single-channel
 */
template<typename T>
void dilate(const Image<T>& input, Image<T>& output, size_t kernel_size);

/**
 * @brief Performs morphological opening on a grayscale image.
 *
 * Opening is erosion followed by dilation. It removes small bright objects
 * and smooths the boundaries of larger bright regions while preserving the
 * overall shape of larger dark regions.
 *
 * @tparam T The data type of the image (typically uint8_t or float)
 * @param input The source grayscale image (must have exactly 1 channel)
 * @param output The destination grayscale image where the result is stored (must have exactly 1 channel)
 * @param kernel_size The size of the square structuring element (must be odd, e.g., 3, 5, 7)
 *
 * @note Input and output images must have the same dimensions and exactly 1 channel.
 *
 * @throws std::invalid_argument If kernel_size is even, input/output dimensions don't match, or images are not single-channel
 *
 * @see erode() for erosion operation
 * @see dilate() for dilation operation
 */
template<typename T>
void open(const Image<T>& input, Image<T>& output, size_t kernel_size);

/**
 * @brief Performs morphological closing on a grayscale image.
 *
 * Closing is dilation followed by erosion. It removes small dark objects
 * (holes) and fills gaps in the boundaries of regions while preserving the
 * overall shape of the image.
 *
 * @tparam T The data type of the image (typically uint8_t or float)
 * @param input The source grayscale image (must have exactly 1 channel)
 * @param output The destination grayscale image where the result is stored (must have exactly 1 channel)
 * @param kernel_size The size of the square structuring element (must be odd, e.g., 3, 5, 7)
 *
 * @note Input and output images must have the same dimensions and exactly 1 channel.
 *
 * @throws std::invalid_argument If kernel_size is even, input/output dimensions don't match, or images are not single-channel
 *
 * @see dilate() for dilation operation
 * @see erode() for erosion operation
 */
template<typename T>
void close(const Image<T>& input, Image<T>& output, size_t kernel_size);

/**
 * @brief Computes the morphological gradient of a grayscale image.
 *
 * The morphological gradient is defined as dilation minus erosion.
 * It highlights the boundaries (edges) of objects in the image, producing
 * a result similar to a edge detection filter.
 *
 * @tparam T The data type of the image (typically uint8_t or float)
 * @param input The source grayscale image (must have exactly 1 channel)
 * @param output The destination grayscale image where the gradient is stored (must have exactly 1 channel)
 * @param kernel_size The size of the square structuring element (must be odd, e.g., 3, 5, 7)
 *
 * @note Input and output images must have the same dimensions and exactly 1 channel.
 *
 * @throws std::invalid_argument If kernel_size is even, input/output dimensions don't match, or images are not single-channel
 *
 * @see dilate() for dilation operation
 * @see erode() for erosion operation
 */
template<typename T>
void morphologicalGradient(const Image<T>& input, Image<T>& output, size_t kernel_size);

/**
 * @brief Performs top-hat transformation on a grayscale image.
 *
 * Top-hat (white top-hat) extracts small bright objects and details from
 * the image by subtracting the opened image from the original. It is useful
 * for enhancing small features that would otherwise be lost during smoothing.
 *
 * @tparam T The data type of the image (typically uint8_t or float)
 * @param input The source grayscale image (must have exactly 1 channel)
 * @param output The destination grayscale image where the result is stored (must have exactly 1 channel)
 * @param kernel_size The size of the square structuring element (must be odd, e.g., 3, 5, 7)
 *
 * @note Input and output images must have the same dimensions and exactly 1 channel.
 *
 * @throws std::invalid_argument If kernel_size is even, input/output dimensions don't match, or images are not single-channel
 *
 * @see open() for the opening operation
 * @see bottomHat() for extracting dark features
 */
template<typename T>
void topHat(const Image<T>& input, Image<T>& output, size_t kernel_size);

/**
 * @brief Performs bottom-hat transformation on a grayscale image.
 *
 * Bottom-hat (black top-hat) extracts small dark objects and holes from
 * the image by subtracting the original image from the closed image. It is
 * useful for enhancing small dark features that would otherwise be lost.
 *
 * @tparam T The data type of the image (typically uint8_t or float)
 * @param input The source grayscale image (must have exactly 1 channel)
 * @param output The destination grayscale image where the result is stored (must have exactly 1 channel)
 * @param kernel_size The size of the square structuring element (must be odd, e.g., 3, 5, 7)
 *
 * @note Input and output images must have the same dimensions and exactly 1 channel.
 *
 * @throws std::invalid_argument If kernel_size is even, input/output dimensions don't match, or images are not single-channel
 *
 * @see close() for the closing operation
 * @see topHat() for extracting bright features
 */
template<typename T>
void bottomHat(const Image<T>& input, Image<T>& output, size_t kernel_size);
