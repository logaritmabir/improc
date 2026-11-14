/**
 * @file pnm_funcs.cpp
 * @brief Implementation of PNM format image I/O.
 *
 * This file contains implementations for reading and writing PNM (Portable
 * Anymap) format images, including support for PGM (grayscale, P5) and PPM
 * (color, P6) formats.
 *
 * @see pnm_funcs.hpp for the function declarations and detailed documentation
 * @note Only binary PNM formats (P5 and P6) are supported; ASCII formats are not implemented
 */

#include "pnm_funcs.hpp"
#include <fstream>
#include <stdexcept>

void readPNM(const std::string& filename, Image<uint8_t>& img, PNMFormat format) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    std::string magic;
    infile >> magic;
    if (!infile) {
        throw std::runtime_error("Invalid PNM header in file: " + filename);
    }

    size_t channels;
    if (format == PNMFormat::P5 && magic == "P5") {
        channels = 1u;
    } else if (format == PNMFormat::P6 && magic == "P6") {
        channels = 3u;
    } else if (format != PNMFormat::P5 && format != PNMFormat::P6) {
        throw std::invalid_argument("Unsupported PNM input format.");
    } else {
        throw std::runtime_error("PNM magic number does not match the requested format in file: " + filename);
    }

    infile.get();
    std::string comment;
    while (infile.peek() == '#') {
        std::getline(infile, comment);
    }

    size_t width, height, maxColorValue;
    infile >> width >> height >> maxColorValue;
    if (!infile) {
        throw std::runtime_error("Invalid PNM header in file: " + filename);
    }
    infile.get();

    if (maxColorValue != 255) {
        throw std::runtime_error("Unsupported max color value (only 255 is supported): " + std::to_string(maxColorValue));
    }

    img = Image<uint8_t>(height, width, channels);
    infile.read(reinterpret_cast<char*>(img.data()), width * height * channels * sizeof(uint8_t));

    if (!infile) {
        throw std::runtime_error("Error reading pixel data from file: " + filename);
    }
}
void savePNM(const std::string& filename, const Image<uint8_t>& img, PNMFormat format) {
    const char* magic;
    size_t channels;
    if (format == PNMFormat::P5) {
        magic = "P5";
        channels = 1u;
    } else if (format == PNMFormat::P6) {
        magic = "P6";
        channels = 3u;
    } else {
        throw std::invalid_argument("Unsupported PNM output format.");
    }

    if (img.channels() != channels) {
        throw std::invalid_argument("Image channel count does not match the requested PNM format.");
    }

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    outfile << magic << "\n" << img.cols() << " " << img.rows() << "\n255\n";
    outfile.write(reinterpret_cast<const char*>(img.data()), img.cols() * img.rows() * img.channels() * sizeof(uint8_t));

    if (!outfile) {
        throw std::runtime_error("Error writing pixel data to file: " + filename);
    }
}
