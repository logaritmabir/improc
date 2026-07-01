/**
 * @file point_ops.cpp
 * @brief Implementation of point-wise image transformation operations.
 *
 * This file contains template implementations for per-pixel operations
 * including image inversion, binary thresholding, and histogram equalization.
 * These operations do not require neighborhood information and can be
 * parallelized efficiently.
 *
 * @see point_ops.hpp for the function declarations and detailed documentation
 */

#include "point_ops.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <array>

template<typename PixelType>
void invert(const Image<PixelType>& input, Image<PixelType>& output) {
    requireSameTypeImages(input, output);
    
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            for(int ch = 0; ch < channels; ch++){
                output(r,c,ch) = static_cast<PixelType>(255 - input(r,c,ch));
            }
        }
    }
}

template<typename PixelType>
void threshold(const Image<PixelType>& input, Image<PixelType>& output, const PixelType thresholdValue) {
    if(input.channels() != 1)
        throw std::invalid_argument("Threshold only works on single-channel (grayscale) images");
    
    if(output.channels() != 1)
        throw std::invalid_argument("Output image must be single-channel (grayscale)");
    
    if(input.rows() != output.rows() || input.cols() != output.cols())
        throw std::invalid_argument("Input and output images must have the same dimensions");
    
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            output(r,c,0) = (input(r,c) >= thresholdValue) ? static_cast<PixelType>(255) : static_cast<PixelType>(0);
        }
    }
}

template<typename PixelType>
void histogramEqualization(const Image<PixelType>& input, Image<PixelType>& output){
    requireSameTypeImages(input, output);
    
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    std::array<int, 256> histogram = {0};

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            histogram[static_cast<int>(input(r,c))]++;
        }
    }

    std::array<int, 256> cdf = {0};
    cdf[0] = histogram[0];
    for(int i = 1; i < 256; i++){
        cdf[i] = cdf[i-1] + histogram[i];
    }

    int totalPixels = rows * cols;
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            output(r,c) = static_cast<PixelType>((cdf[static_cast<int>(input(r,c))] - cdf[0]) * 255 / (totalPixels - cdf[0]));
        }
    }
}

template void invert(const Image<uint8_t>& input, Image<uint8_t>& output);
template void threshold(const Image<uint8_t>& input, Image<uint8_t>& output, const uint8_t thresholdValue);
template void histogramEqualization(const Image<uint8_t>& input, Image<uint8_t>& output);
