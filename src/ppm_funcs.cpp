#include "ppm_funcs.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

template<typename T>
void readPPM(const std::string& filename, Image<T>& img) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    std::string magic;
    infile >> magic;
    
    bool isGrayscale = (magic == "P5");
    bool isColor = (magic == "P6");
    
    if (!isGrayscale && !isColor) {
        throw std::runtime_error("Unsupported PPM format (only P5 and P6 are supported): " + magic);
    }

    size_t width, height;
    size_t maxColorValue;

    infile >> width >> height >> maxColorValue;
    infile.get(); // consume the newline after maxColorValue

    if (maxColorValue != 255) {
        throw std::runtime_error("Unsupported max color value (only 255 is supported): " + std::to_string(maxColorValue));
    }

    uint32_t channels = isGrayscale ? 1 : 3;
    img = Image<T>(height, width, channels);

    infile.read(reinterpret_cast<char*>(img.data()), width * height * channels);

    if (!infile) {
        throw std::runtime_error("Error reading pixel data from file: " + filename);
    }

    infile.close();
}

template<typename T>
void savePPM(const std::string& filename, Image<T>& img) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    std::string magic = (img.channels() == 1) ? "P5" : "P6";
    outfile << magic << "\n" << img.cols() << " " << img.rows() << "\n255\n";
    outfile.write(reinterpret_cast<char*>(img.data()), img.cols() * img.rows() * img.channels());

    if (!outfile) {
        throw std::runtime_error("Error writing pixel data to file: " + filename);
    }

    outfile.close();
}

template void readPPM<uint8_t>(const std::string& filename, Image<uint8_t>& img);
template void savePPM<uint8_t>(const std::string& filename, Image<uint8_t>& img);