/**
 * @file pnm_funcs.cpp
 * @brief Implementation of PNM format image I/O and color processing.
 *
 * This file contains implementations for reading and writing PNM (Portable
 * Anymap) format images, including support for PGM (grayscale, P5) and PPM
 * (color, P6) formats. Also includes color space conversion functions
 * (RGB to grayscale) and channel extraction utilities.
 *
 * @see pnm_funcs.hpp for the function declarations and detailed documentation
 * @note Only binary PNM formats (P5 and P6) are supported; ASCII formats are not implemented
 */

#include "pnm_funcs.hpp"
#include <fstream>
#include <stdexcept>

template<typename PixelType>
void readPNM(const std::string& filename, Image<PixelType>& img, PNMFormat format) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    std::string magic;
    infile >> magic;
    if (!infile) {
        throw std::runtime_error("Invalid PNM header in file: " + filename);
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

    size_t channels;
    if (format == PNMFormat::P5) {
        channels = 1u;
    } else if (format == PNMFormat::P6) {
        channels = 3u;
    } else {
        throw std::invalid_argument("Unsupported PNM input format.");
    }

    img = Image<PixelType>(height, width, channels);
    infile.read(reinterpret_cast<char*>(img.data()), width * height * channels * sizeof(PixelType));

    if (!infile) {
        throw std::runtime_error("Error reading pixel data from file: " + filename);
    }
}

template<typename PixelType>
void savePNM(const std::string& filename, const Image<PixelType>& img, PNMFormat format) {
    const char* magic;
    if (format == PNMFormat::P5) {
        magic = "P5";
    } else if (format == PNMFormat::P6) {
        magic = "P6";
    } else {
        throw std::invalid_argument("Unsupported PNM output format.");
    }

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    outfile << magic << "\n" << img.cols() << " " << img.rows() << "\n255\n";
    outfile.write(reinterpret_cast<const char*>(img.data()), img.cols() * img.rows() * img.channels() * sizeof(PixelType));

    if (!outfile) {
        throw std::runtime_error("Error writing pixel data to file: " + filename);
    }
}

template<typename PixelType>
void extractChannel(const Image<PixelType>& input, Image<PixelType>& output, size_t channel) {
    if (input.channels() != 3) {
        throw std::invalid_argument("Image must be RGB (3 channels) to extract a channel");
    }
    if (channel > 2) {
        throw std::invalid_argument("Channel must be 0 (R), 1 (G), or 2 (B)");
    }
    
    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t col = 0; col < input.cols(); col++) {
            output.at(row, col, 0) = input.at(row, col, channel);
        }
    }
}

template<typename PixelType>
void rgbToGrayscale(const Image<PixelType>& input, Image<PixelType>& output) {
    if (input.channels() != 3) {
        throw std::invalid_argument("Image must be RGB (3 channels)");
    }
    
    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t col = 0; col < input.cols(); col++) {
            PixelType r = input.at(row, col, 0);
            PixelType g = input.at(row, col, 1);
            PixelType b = input.at(row, col, 2);
            
            PixelType gray = static_cast<PixelType>(0.299 * r + 0.587 * g + 0.114 * b);
            output.at(row, col, 0) = gray;
        }
    }
}

template void readPNM<uint8_t>(const std::string& filename, Image<uint8_t>& img, PNMFormat format);
template void savePNM<uint8_t>(const std::string& filename, const Image<uint8_t>& img, PNMFormat format);
template void extractChannel<uint8_t>(const Image<uint8_t>& input, Image<uint8_t>& output, size_t channel);
template void rgbToGrayscale<uint8_t>(const Image<uint8_t>& input, Image<uint8_t>& output);
