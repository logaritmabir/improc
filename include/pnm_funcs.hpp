#pragma once

#include "image.hpp"
#include <string>

/**
 * @file pnm_funcs.hpp
 * @brief PNM image I/O and color processing functions.
 *
 * Supports 8-bit binary PGM (P5) and PPM (P6) image files.
 */

enum class PNMFormat {
    P5,
    P6
};

/**
 * @brief Reads a P5 or P6 image selected explicitly by the caller.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path to the PNM file
 * @param img Image object to store the loaded data
 * @param format Required PNM variant
 * @throws std::runtime_error If the file cannot be opened or its format is unsupported
 */
template<typename PixelType>
void readPNM(const std::string& filename, Image<PixelType>& img, PNMFormat format);

/**
 * @brief Saves an image in the explicitly selected P5 or P6 format.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path where the image will be saved
 * @param img Single- or three-channel image to save
 * @param format Required PNM variant
 * @throws std::invalid_argument If the image channels do not match the requested format
 * @throws std::runtime_error If the file cannot be opened or written
 */
template<typename PixelType>
void savePNM(const std::string& filename, const Image<PixelType>& img, PNMFormat format);

/**
 * @brief Extracts one color channel from a three-channel RGB image.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param input Source RGB image
 * @param output Destination single-channel image
 * @param channel Channel index: 0=Red, 1=Green, 2=Blue
 */
template<typename PixelType>
void extractChannel(const Image<PixelType>& input, Image<PixelType>& output, size_t channel);

/**
 * @brief Converts a three-channel RGB image to grayscale.
 *
 * Uses Gray = 0.299*R + 0.587*G + 0.114*B.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param input Source RGB image
 * @param output Destination single-channel image
 */
template<typename PixelType>
void rgbToGrayscale(const Image<PixelType>& input, Image<PixelType>& output);
