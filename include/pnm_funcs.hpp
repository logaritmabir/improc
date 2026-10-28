#pragma once

#include "image.hpp"
#include <string>

/**
 * @file pnm_funcs.hpp
 * @brief PNM image I/O and color processing functions.
 *
 * Supports binary PGM (P5) and PPM (P6) image files.
 */

/**
 * @brief Reads a binary PGM (P5) or PPM (P6) image file.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path to the image file
 * @param img Image object to store the loaded data
 * @throws std::runtime_error If the file cannot be opened or the format is unsupported
 */
template<typename PixelType>
void readPNM(const std::string& filename, Image<PixelType>& img);

/**
 * @brief Saves an image as binary PGM (P5) or PPM (P6).
 *
 * A single-channel image is saved as P5; other channel counts are currently
 * written with a P6 header.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path where the image will be saved
 * @param img Image object to save
 * @throws std::runtime_error If the file cannot be opened or written
 */
template<typename PixelType>
void savePNM(const std::string& filename, const Image<PixelType>& img);

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
