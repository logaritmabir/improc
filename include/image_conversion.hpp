#pragma once

/**
 * @file image_conversion.hpp
 * @brief Pixel-type and color-layout conversions for images.
 *
 * This module contains image transformations independent of file formats,
 * including channel extraction, RGB-to-grayscale conversion, and pixel-value
 * range conversion.
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

/**
 * @brief Converts each pixel to a different pixel type.
 *
 * Applies `static_cast<DestinationPixelType>` to each pixel value. Values are
 * not normalized or clamped; callers must ensure that values are representable
 * by the destination type.
 *
 * @tparam SourcePixelType Source image pixel type
 * @tparam DestinationPixelType Destination image pixel type
 * @param input Source image
 * @param output Destination image; must match the source dimensions and channel count
 * @throws std::invalid_argument If the image dimensions or channel counts differ
 */
template<typename SourcePixelType, typename DestinationPixelType>
void convertImageType(const Image<SourcePixelType>& input, Image<DestinationPixelType>& output);

/**
 * @brief Converts pixel values from one explicit numeric range to another.
 *
 * Values outside the source range are clamped to that range before mapping.
 * Floating-point destinations preserve fractional results. Integer
 * destinations are rounded to the nearest value.
 *
 * @warning This function was written by an AI system and has not been
 * independently reviewed or verified by a human. Review its implementation
 * before relying on it.
 *
 * @tparam SourcePixelType Source image pixel type
 * @tparam DestinationPixelType Destination image pixel type
 * @param input Source image
 * @param output Destination image; must match the source dimensions and channels
 * @param sourceMin Minimum value of the source range
 * @param sourceMax Maximum value of the source range
 * @param destinationMin Minimum value of the destination range
 * @param destinationMax Maximum value of the destination range
 * @throws std::invalid_argument If either range is non-finite or not ascending,
 * image metadata differs, a destination bound is not representable, or a NaN
 * source value is converted to an integer type
 */
template<typename SourcePixelType, typename DestinationPixelType>
void convertImageRange(
    const Image<SourcePixelType>& input,
    Image<DestinationPixelType>& output,
    long double sourceMin,
    long double sourceMax,
    long double destinationMin,
    long double destinationMax);

#include "image_conversion.tpp"
