#include "filters.hpp"
#include "utils.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <array>

// sep conva bakılacak
// validate kernel fonksiyonuna baklıacak, nerelerde kullanılabilir vb.

template<typename T>
void convolve(const Image<T>& input, Image<T>& output, const std::vector<std::vector<float>>& kernel) {
    if(kernel.empty())
        throw std::invalid_argument("Kernel cannot be empty");
    
    size_t kernelSize = kernel.size();

    for(const auto& row : kernel) {
        if(row.size() != kernelSize)
            throw std::invalid_argument("Kernel must be square");
    }

    if(kernelSize % 2 == 0) 
        throw std::invalid_argument("Kernel size must be odd");

    // This line will be replaced with operator!= (maybe) once it's implemented, but for now we can just check dimensions here.
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");

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
                output(r,c,ch) = static_cast<T>(clamp<float>(sum, 0.0f, 255.0f));
            }
        }
    }
}

// Template specialization for std::array kernels
template<typename T, size_t N>
void convolve(const Image<T>& input, Image<T>& output, const std::array<std::array<float, N>, N>& kernel) {
    static_assert(N % 2 == 1, "Kernel size must be odd");
    
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");

    int filterRadius = static_cast<int>(N / 2);
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
                output(r,c,ch) = static_cast<T>(clamp<float>(sum, 0.0f, 255.0f));
            }
        }
    }
}

template<typename T>
void medianFilter(const Image<T>& input, Image<T>& output, size_t kernelSize) {
    if(kernelSize <= 1)
        throw std::invalid_argument("Kernel size must be greater than 1 and odd");
    
    if(kernelSize % 2 == 0) 
        throw std::invalid_argument("Kernel size must be odd");

    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");

    int filterRadius = static_cast<int>(kernelSize / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    std::vector<T> neighbors;
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

template<typename T>
void gaussianBlur(const Image<T>& input, Image<T>& output, float sigma) {
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

    // Normalize the kernel
    for(int i = 0; i < kernelSize; i++){
        for(int j = 0; j < kernelSize; j++){
            kernel[i][j] /= sum;
        }
    }
    convolve(input, output, kernel);
}

template<typename T>
void gaussianBlur(const Image<T>& input, Image<T>& output){
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");

    constexpr std::array<std::array<float, 3>, 3> kernel = {{
        {{0.0625f, 0.1250f, 0.0625f}},
        {{0.1250f, 0.2500f, 0.1250f}},
        {{0.0625f, 0.1250f, 0.0625f}}
    }};

    convolve(input, output, kernel);
}

template<typename T>
void sobelFilter(const Image<T>& input, Image<T>& output) {
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");
    
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

    convolve(input, output, Gx);
    convolve(input, output, Gy);
}

template<typename T>
void sharpeningFilter(const Image<T>& input, Image<T>& output){
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");

    constexpr std::array<std::array<float, 3>, 3> kernel = {{
        {{0.0f, -1.0f, 0.0f}},
        {{-1.0f, 5.0f, -1.0f}},
        {{0.0f, -1.0f, 0.0f}}
    }};

    convolve(input, output, kernel);
}

template<typename T>
void laplacianFilter(const Image<T>& input, Image<T>& output){
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");

    constexpr std::array<std::array<float, 3>, 3> kernel = {{
        {{0.0f, 1.0f, 0.0f}},
        {{1.0f, -4.0f, 1.0f}},
        {{0.0f, 1.0f, 0.0f}}
    }};

    convolve(input, output, kernel);
}

template void medianFilter(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernelSize);
template void convolve(const Image<uint8_t>& input, Image<uint8_t>& output, const std::vector<std::vector<float>>& kernel);
template void convolve(const Image<uint8_t>& input, Image<uint8_t>& output, const std::array<std::array<float, 3>, 3>& kernel);
template void gaussianBlur(const Image<uint8_t>& input, Image<uint8_t>& output, float sigma);
template void gaussianBlur(const Image<uint8_t>& input, Image<uint8_t>& output);
template void sobelFilter(const Image<uint8_t>& input, Image<uint8_t>& output);
template void sharpeningFilter(const Image<uint8_t>& input, Image<uint8_t>& output);
template void laplacianFilter(const Image<uint8_t>& input, Image<uint8_t>& output);
