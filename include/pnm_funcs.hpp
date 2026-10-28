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
    P5,
    P6
};

/**
 * @brief Reads a P5 or P6 image selected explicitly by the caller.
 *
 * @tparam PixelType Must be uint8_t
 * @param filename Path to the PNM file
 * @param img Image object to store the loaded data
 * @param format Required PNM variant
 * @throws std::runtime_error If the file cannot be opened or its header/pixel data is invalid
 * @throws std::invalid_argument If the requested format is not P5 or P6
 */
template<typename PixelType>
void readPNM(const std::string& filename, Image<PixelType>& img, PNMFormat format);

/**
 * @brief Saves an image in the explicitly selected P5 or P6 format.
 *
 * @tparam PixelType Must be uint8_t
 * @param filename Path where the image will be saved
 * @param img Image to save; its channel layout must match the selected format
 * @param format Required PNM variant
 * @throws std::runtime_error If the file cannot be opened or written
 */
template<typename PixelType>
void savePNM(const std::string& filename, const Image<PixelType>& img, PNMFormat format);

/**
 * @brief Extracts one color channel from a three-channel RGB image.
 *
 * @tparam PixelType Must be uint8_t
 * @param input Source RGB image
 * @param output Destination single-channel image
 * @param channel Channel index: 0=Red, 1=Green, 2=Blue
 * @throws std::invalid_argument If the input is not RGB or the channel is invalid
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
 * @param output Destination single-channel image
 * @throws std::invalid_argument If the input is not RGB
 */
template<typename PixelType>
void rgbToGrayscale(const Image<PixelType>& input, Image<PixelType>& output);
