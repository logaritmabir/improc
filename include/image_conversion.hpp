#pragma once

/**
 * @file image_conversion.hpp
 * @brief Pixel-type and color-layout conversions for images.
 *
 * This module contains image transformations independent of file formats,
 * including channel extraction and RGB-to-grayscale conversion.
 */

#include <cstddef>
#include <stdexcept>
#include "image.hpp"

/**
 * @brief Extracts one color channel from a three-channel RGB image.
 *
 * @tparam PixelType Image pixel type
 * @param input Source RGB image
 * @param output Destination image; channel 0 receives the extracted values
 * @param channel Channel index: 0=Red, 1=Green, 2=Blue
 * @throws std::invalid_argument If the input is not RGB or the channel is invalid
 * @throws std::out_of_range If @p output is too small for the extracted channel
 *
 * @pre @p output must have the same width and height as @p input and at least
 * one channel. Other output channels, if present, are left unchanged.
 */
template<typename PixelType>
void extractChannel(const Image<PixelType>& input, Image<PixelType>& output, size_t channel);

/**
 * @brief Converts a three-channel RGB image to grayscale.
 *
 * Uses Gray = 0.299*R + 0.587*G + 0.114*B.
 *
 * @tparam PixelType Image pixel type
 * @param input Source RGB image
 * @param output Destination image; channel 0 receives the grayscale values
 * @throws std::invalid_argument If the input is not RGB
 * @throws std::out_of_range If @p output is too small for the grayscale result
 *
 * @pre @p output must have the same width and height as @p input and at least
 * one channel. Other output channels, if present, are left unchanged.
 */
template<typename PixelType>
void rgbToGrayscale(const Image<PixelType>& input, Image<PixelType>& output);

#include "image_conversion.tpp"
