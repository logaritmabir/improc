/**
 * @file morph.cpp
 * @brief Implementation of morphological image processing operations.
 *
 * This file contains template implementations for morphological operations
 * including erosion, dilation, opening, closing, morphological gradient,
 * and top/bottom-hat transformations. These operations are fundamental
 * to image processing for shape analysis and feature extraction.
 *
 * @see morph.hpp for the function declarations and detailed documentation
 * @note All morphological operations require single-channel (grayscale) images
 */

#include "morph.hpp"
#include "utils.hpp"
#include <limits>

template<typename PixelType>
void erode(const Image<PixelType>& input, Image<PixelType>& output, size_t kernel_size){
    requireOddSize(kernel_size);

    if(input.channels() != 1)
        throw std::invalid_argument("Erosion only works on single-channel (grayscale) images");
    
    if(output.channels() != 1)
        throw std::invalid_argument("Output image must be single-channel (grayscale)");
    
    if(input.rows() != output.rows() || input.cols() != output.cols())
        throw std::invalid_argument("Input and output images must have the same dimensions");

    int filterRadius = static_cast<int>(kernel_size / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            PixelType minVal = std::numeric_limits<PixelType>::max();

            for(int i = -filterRadius; i <= filterRadius; ++i) {
                for(int j = -filterRadius; j <= filterRadius; ++j) {
                    int neighborRow = r + i;
                    int neighborCol = c + j;

                    if(neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        PixelType pixelVal = input(neighborRow, neighborCol, 0);
                        minVal = std::min(minVal, pixelVal);
                    }
                }
            }

            output(r, c, 0) = minVal;
        }
    }
}

template<typename PixelType>
void dilate(const Image<PixelType>& input, Image<PixelType>& output, size_t kernel_size){
    requireOddSize(kernel_size);

    if(input.channels() != 1)
        throw std::invalid_argument("Dilation only works on single-channel (grayscale) images");
    
    if(output.channels() != 1)
        throw std::invalid_argument("Output image must be single-channel (grayscale)");
    
    if(input.rows() != output.rows() || input.cols() != output.cols())
        throw std::invalid_argument("Input and output images must have the same dimensions");

    int filterRadius = static_cast<int>(kernel_size / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            PixelType maxVal = std::numeric_limits<PixelType>::min();
            for(int i = -filterRadius; i <= filterRadius; ++i) {
                for(int j = -filterRadius; j <= filterRadius; ++j) {
                    int neighborRow = r + i;
                    int neighborCol = c + j;

                    if(neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        PixelType pixelVal = input(neighborRow, neighborCol, 0);
                        maxVal = std::max(maxVal, pixelVal);
                    }
                }
            }

            output(r, c, 0) = maxVal;
        }
    }
}

template<typename PixelType>
void open(const Image<PixelType>& input, Image<PixelType>& output, size_t kernel_size){
    Image<PixelType> temp(output);
    erode(input, temp, kernel_size);
    dilate(temp, output, kernel_size);
}

template<typename PixelType>
void close(const Image<PixelType>& input, Image<PixelType>& output, size_t kernel_size){
    Image<PixelType> temp(output);
    dilate(input, temp, kernel_size);
    erode(temp, output, kernel_size);
}

template<typename PixelType>
void morphologicalGradient(const Image<PixelType>& input, Image<PixelType>& output, size_t kernel_size){
    Image<PixelType> dilated(input.rows(), input.cols(), input.channels());
    Image<PixelType> eroded(input.rows(), input.cols(), input.channels());

    dilate(input, dilated, kernel_size);
    erode(input, eroded, kernel_size);

    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            output(r, c) = dilated(r, c) - eroded(r, c);
        }
    }
}

template<typename PixelType>
void topHat(const Image<PixelType>& input, Image<PixelType>& output, size_t kernel_size){
    Image<PixelType> opened(input.rows(), input.cols(), input.channels());
    open(input, opened, kernel_size);

    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            output(r, c) = input(r, c) - opened(r, c);
        }
    }
}

template<typename PixelType>
void bottomHat(const Image<PixelType>& input, Image<PixelType>& output, size_t kernel_size){
    Image<PixelType> closed(input.rows(), input.cols(), input.channels());
    close(input, closed, kernel_size);

    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            output(r, c) = closed(r, c) - input(r, c);
        }
    }
}

template void erode(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
template void dilate(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
template void open(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
template void close(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
template void morphologicalGradient(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
template void topHat(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
template void bottomHat(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);

    
