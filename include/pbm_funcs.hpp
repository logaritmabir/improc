#pragma once

#include "image.hpp"
#include <string>

/**
 * @file pnm_funcs.hpp
 * @brief PNM (Portable PixMap/GrayMap) image I/O and color processing functions.
 *
 * This module provides functions for reading and writing PNM format images only.
 * Supported formats: PGM (P5 - grayscale) and PPM (P6 - RGB color).
 */

/**
 * @brief Reads a PNM image file (P5 grayscale or P6 color).
 *
 * @tparam T Data type for pixel values (typically uint8_t)
 * @param filename Path to the PNM file
 * @param img Image object to store the loaded data
 * @throws std::runtime_error If file cannot be opened or format is not P5/P6
 *
 * @note Only PNM format (P5 and P6) is supported.
 */
template<typename T>
void readPNM(const std::string& filename, Image<T>& img);

/**
 * @brief Saves an image to a PNM file (P5 for grayscale, P6 for color).
 *
 * @tparam T Data type for pixel values (typically uint8_t)
 * @param filename Path where the PNM file will be saved
 * @param img Image object to save
 * @throws std::runtime_error If file cannot be opened for writing
 *
 * @note Only PNM format output is supported. Output format (P5/P6) is determined by the image's channel count.
 */
template<typename T>
void savePNM(const std::string& filename, const Image<T>& img);

/**
 * @brief Extracts a single color channel from an RGB image.
 *
 * @tparam T Data type for pixel values (typically uint8_t)
 * @param input RGB image (3 channels) in PNM format
 * @param output Output grayscale image (1 channel)
 * @param channel Channel index: 0=Red, 1=Green, 2=Blue
 * @throws std::runtime_error If input is not RGB (3 channels)
 *
 * @note Input must be loaded from PNM format (PPM color image).
 */
template<typename T>
void extractChannel(const Image<T>& input, Image<T>& output, size_t channel);

/**
 * @brief Converts an RGB image to grayscale using standard luminosity formula.
 *
 * Uses the formula: Gray = 0.299*R + 0.587*G + 0.114*B
 *
 * @tparam T Data type for pixel values (typically uint8_t)
 * @param input RGB image (3 channels) in PNM format
 * @param output Output grayscale image (1 channel)
 * @throws std::runtime_error If input is not RGB (3 channels)
 *
 * @note Input must be loaded from PNM format (PPM color image).
 */
template<typename T>
void rgbToGrayscale(const Image<T>& input, Image<T>& output);
