#pragma once

#include "image.hpp"
#include <cstdint>
#include <string>

/**
 * @file pnm_funcs.hpp
 * @brief PNM image I/O and color processing functions.
 *
 * Supports only uint8_t images in 8-bit binary PGM (P5) and PPM (P6)
 * formats. Other PixelType values are not supported by this module.
 */

enum class PNMFormat {
    /// Binary grayscale PNM (PGM).
    P5,
    /// Binary RGB PNM (PPM).
    P6
};

/**
 * @brief Reads a P5 or P6 image selected explicitly by the caller.
 *
 * The selected format determines the number of channels used to interpret
 * the payload. The file's magic number is not checked against @p format.
 * Only 8-bit samples with maxval 255 are supported.
 *
 * @tparam PixelType Must be uint8_t
 * @param filename Path to the PNM file
 * @param img Image object to store the loaded data
 * @param format Required PNM variant
 * @throws std::runtime_error If the file cannot be opened, the header is
 * invalid, maxval is unsupported, or pixel data cannot be read
 * @throws std::invalid_argument If @p format is not P5 or P6, or an image
 * dimension is zero
 */
template<typename PixelType>
void readPNM(const std::string& filename, Image<PixelType>& img, PNMFormat format);

/**
 * @brief Saves an image in the explicitly selected P5 or P6 format.
 *
 * @tparam PixelType Must be uint8_t
 * @param filename Path where the image will be saved
 * @param img Image to save; its channel layout must match the selected format
 * (one channel for P5, three for P6)
 * @param format Required PNM variant
 * @throws std::runtime_error If the file cannot be opened or written
 * @throws std::invalid_argument If @p format is not P5 or P6
 *
 * @note The channel layout is a caller precondition and is not validated.
 */
template<typename PixelType>
void savePNM(const std::string& filename, const Image<PixelType>& img, PNMFormat format);

/**
 * @brief Extracts one color channel from a three-channel RGB image.
 *
 * @tparam PixelType Must be uint8_t
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
 * @tparam PixelType Must be uint8_t
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
