#pragma once

#include "image.hpp"
#include <cstdint>
#include <string>

/**
 * @file pnm_funcs.hpp
 * @brief PNM image input and output functions.
 *
 * Supports only uint8_t images in 8-bit binary PGM (P5) and PPM (P6)
 * formats.
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
 * The selected format must match the file's magic number and determines the
 * number of channels used to interpret the payload. Only 8-bit samples with
 * maxval 255 are supported.
 *
 * @param filename Path to the PNM file
 * @param img Image object to store the loaded uint8_t data
 * @param format Required PNM variant
 * @throws std::runtime_error If the file cannot be opened, the header is
 * invalid, the magic number does not match @p format, maxval is unsupported,
 * or pixel data cannot be read
 * @throws std::invalid_argument If @p format is not P5 or P6, or an image
 * dimension is zero
 */
void readPNM(const std::string& filename, Image<uint8_t>& img, PNMFormat format);

/**
 * @brief Saves an image in the explicitly selected P5 or P6 format.
 *
 * @param filename Path where the image will be saved
 * @param img uint8_t image to save; its channel layout must match the selected format
 * (one channel for P5, three for P6)
 * @param format Required PNM variant
 * @throws std::runtime_error If the file cannot be opened or written
 * @throws std::invalid_argument If @p format is not P5 or P6, or the image
 * channel count does not match the selected format
 */
void savePNM(const std::string& filename, const Image<uint8_t>& img, PNMFormat format);
