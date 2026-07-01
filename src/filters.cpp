/**
 * @file filters.cpp
 * @brief Implementation of image filtering and convolution operations.
 *
 * This file contains template implementations for various spatial filters,
 * including convolution, median filtering, Gaussian blur, edge detection
 * (Sobel), and sharpening. All CPU-side filtering operations are implemented
 * as template instantiations to support multiple data types.
 *
 * @see filters.hpp for the function declarations and detailed documentation
 */

#include "filters.hpp"
#include "utils.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <array>

template<typename PixelType>
void convolve(const Image<PixelType>& input, Image<PixelType>& output, const std::vector<std::vector<float>>& kernel) {
    requireSquareAndOdd(kernel);
    requireSameTypeImages(input, output);

    size_t kernelSize = kernel.size();

    int filterRadius = static_cast<int>(kernelSize / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            for(int ch = 0; ch < channels; ch++){
                float sum = 0.0f;
                for(int i = -filterRadius; i <= filterRadius; i++){
                    for(int j = -filterRadius; j <= filterRadius; j++){
                        int row = r + i;
                        int col = c + j;

                        if(row >= 0 && row < rows && col >= 0 && col < cols)
                            sum += input(row, col, ch) * kernel[i + filterRadius][j + filterRadius];
                    }
                }
                output(r,c,ch) = static_cast<PixelType>(clamp<float>(sum, 0.0f, 255.0f));
            }
        }
    }
}

template<typename PixelType, size_t KernelSize>
void convolve(const Image<PixelType>& input, Image<PixelType>& output, const std::array<std::array<float, KernelSize>, KernelSize>& kernel) {
    requireSquareAndOdd(kernel);
    requireSameTypeImages(input, output);

    int filterRadius = static_cast<int>(KernelSize / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            for(int ch = 0; ch < channels; ch++){
                float sum = 0.0f;
                for(int i = -filterRadius; i <= filterRadius; i++){
                    for(int j = -filterRadius; j <= filterRadius; j++){
                        int row = r + i;
                        int col = c + j;

                        if(row >= 0 && row < rows && col >= 0 && col < cols)
                            sum += input(row, col, ch) * kernel[i + filterRadius][j + filterRadius];
                    }
                }
                output(r,c,ch) = static_cast<PixelType>(clamp<float>(sum, 0.0f, 255.0f));
            }
        }
    }
}

template<typename PixelType>
void medianFilter(const Image<PixelType>& input, Image<PixelType>& output, size_t kernelSize) {
    if(kernelSize <= 1)
        throw std::invalid_argument("Kernel size must be greater than 1 and odd");

    requireOddSize(kernelSize);
    requireSameTypeImages(input, output);

    int filterRadius = static_cast<int>(kernelSize / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    std::vector<PixelType> neighbors;
    neighbors.reserve(kernelSize * kernelSize); // Reserve space for neighbors to avoid reallocations

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            for(int ch = 0; ch < channels; ch++){
                neighbors.clear(); // Clear the neighbors vector for the new pixel
                for(int i = -filterRadius; i <= filterRadius; i++){
                    for(int j = -filterRadius; j <= filterRadius; j++){
                        int row = r + i;
                        int col = c + j;

                        if(row >= 0 && row < rows && col >= 0 && col < cols){
                            neighbors.push_back(input(row, col, ch));
                        }
                    }
                }
                std::sort(neighbors.begin(), neighbors.end());
                output(r,c,ch) = neighbors[neighbors.size() / 2]; // Set to median value
            }
        }
    }
}

template<typename PixelType>
void gaussianBlur(const Image<PixelType>& input, Image<PixelType>& output, float sigma) {
    if(sigma <= 0.0f)
        throw std::invalid_argument("Sigma must be greater than 0");

    int kernelSize = static_cast<int>(std::ceil(sigma * 6)) | 1; // Ensure kernel size is odd
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    int filterRadius = kernelSize / 2;
    float sum = 0.0f;

    for(int i = -filterRadius; i <= filterRadius; i++){
        for(int j = -filterRadius; j <= filterRadius; j++){
            float exponent = -(i * i + j * j) / (2 * sigma * sigma);
            kernel[i + filterRadius][j + filterRadius] = std::exp(exponent);
            sum += kernel[i + filterRadius][j + filterRadius];
        }
    }

    for(int i = 0; i < kernelSize; i++){
        for(int j = 0; j < kernelSize; j++){
            kernel[i][j] /= sum;
        }
    }
    convolve(input, output, kernel);
}

template<typename PixelType>
void gaussianBlur(const Image<PixelType>& input, Image<PixelType>& output){
    requireSameTypeImages(input, output);

    constexpr std::array<std::array<float, 3>, 3> kernel = {{
        {{0.0625f, 0.1250f, 0.0625f}},
        {{0.1250f, 0.2500f, 0.1250f}},
        {{0.0625f, 0.1250f, 0.0625f}}
    }};

    convolve(input, output, kernel);
}

template<typename PixelType>
void sobelFilter(const Image<PixelType>& input, Image<PixelType>& output) {
    requireSameTypeImages(input, output);

    constexpr std::array<std::array<float, 3>, 3> Gx = {{
        {{-1.0f, 0.0f, 1.0f}},
        {{-2.0f, 0.0f, 2.0f}},
        {{-1.0f, 0.0f, 1.0f}}
    }};

    constexpr std::array<std::array<float, 3>, 3> Gy = {{
        {{1.0f, 2.0f, 1.0f}},
        {{0.0f, 0.0f, 0.0f}},
        {{-1.0f, -2.0f, -1.0f}}
    }};

    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            for(int ch = 0; ch < channels; ch++){
                float sumX = 0.0f;
                float sumY = 0.0f;

                for(int i = -1; i <= 1; i++){
                    for(int j = -1; j <= 1; j++){
                        int row = r + i;
                        int col = c + j;

                        if(row >= 0 && row < rows && col >= 0 && col < cols) {
                            const float pixel = static_cast<float>(input(row, col, ch));
                            sumX += pixel * Gx[i + 1][j + 1];
                            sumY += pixel * Gy[i + 1][j + 1];
                        }
                    }
                }

                const float magnitude = std::sqrt(sumX * sumX + sumY * sumY);
                output(r, c, ch) = static_cast<PixelType>(clamp<float>(magnitude, 0.0f, 255.0f));
            }
        }
    }
}

template<typename PixelType>
void sharpeningFilter(const Image<PixelType>& input, Image<PixelType>& output){
    requireSameTypeImages(input, output);

    constexpr std::array<std::array<float, 3>, 3> kernel = {{
        {{0.0f, -1.0f, 0.0f}},
        {{-1.0f, 5.0f, -1.0f}},
        {{0.0f, -1.0f, 0.0f}}
    }};

    convolve(input, output, kernel);
}

template<typename PixelType>
void laplacianFilter(const Image<PixelType>& input, Image<PixelType>& output){
    requireSameTypeImages(input, output);

    constexpr std::array<std::array<float, 3>, 3> kernel = {{
        {{0.0f, 1.0f, 0.0f}},
        {{1.0f, -4.0f, 1.0f}},
        {{0.0f, 1.0f, 0.0f}}
    }};

    convolve(input, output, kernel);
}

template void medianFilter(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernelSize);
template void convolve(const Image<int>& input, Image<int>& output, const std::array<std::array<float, 3>, 3>& kernel);
template void convolve(const Image<uint8_t>& input, Image<uint8_t>& output, const std::vector<std::vector<float>>& kernel);
template void convolve(const Image<uint8_t>& input, Image<uint8_t>& output, const std::array<std::array<float, 3>, 3>& kernel);
template void gaussianBlur(const Image<uint8_t>& input, Image<uint8_t>& output, float sigma);
template void gaussianBlur(const Image<uint8_t>& input, Image<uint8_t>& output);
template void sobelFilter(const Image<uint8_t>& input, Image<uint8_t>& output);
template void sharpeningFilter(const Image<uint8_t>& input, Image<uint8_t>& output);
template void laplacianFilter(const Image<uint8_t>& input, Image<uint8_t>& output);
