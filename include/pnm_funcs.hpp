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
 * @brief Reads an 8-bit binary grayscale PGM (P5) image.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path to the P5 file
 * @param img Image object to store the loaded data
 * @throws std::runtime_error If the file cannot be opened or is not a supported P5 file
 */
template<typename PixelType>
void readPgmP5(const std::string& filename, Image<PixelType>& img);

/**
 * @brief Reads an 8-bit binary RGB PPM (P6) image.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path to the P6 file
 * @param img Image object to store the loaded data
 * @throws std::runtime_error If the file cannot be opened or is not a supported P6 file
 */
template<typename PixelType>
void readPpmP6(const std::string& filename, Image<PixelType>& img);

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
 * @brief Saves a single-channel image as an 8-bit binary PGM (P5) file.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path where the P5 file will be saved
 * @param img Single-channel image to save
 * @throws std::invalid_argument If the image does not have one channel
 * @throws std::runtime_error If the file cannot be opened or written
 */
template<typename PixelType>
void savePgmP5(const std::string& filename, const Image<PixelType>& img);

/**
 * @brief Saves a three-channel image as an 8-bit binary RGB PPM (P6) file.
 *
 * @tparam PixelType Data type for pixel values (typically uint8_t)
 * @param filename Path where the P6 file will be saved
 * @param img Three-channel image to save
 * @throws std::invalid_argument If the image does not have three channels
 * @throws std::runtime_error If the file cannot be opened or written
 */
template<typename PixelType>
void savePpmP6(const std::string& filename, const Image<PixelType>& img);

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
