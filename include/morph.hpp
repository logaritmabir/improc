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