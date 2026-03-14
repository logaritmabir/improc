#include "filters.hpp"
#include <iostream>

template<typename T>
void convolve(const Image<T>& input, Image<T>& output, const std::vector<std::vector<T>>& kernel, T divideBy) {
    size_t filterSize = kernel.size();
    size_t offset = filterSize / 2;
    
    size_t cols = input.cols();
    size_t rows = input.rows();

    for(size_t r = 0; r < rows; r++){
        for(size_t c = 0; c < cols; c++){
            for(size_t ch = 0; ch < input.channels(); ch++){
                size_t sum = 0;
                for(size_t i = 0; i < filterSize; i++){
                    for(size_t j = 0; j < filterSize; j++){
                        size_t row = r + i - offset;
                        size_t col = c + j - offset;

                        if(row >= 0 && row < rows && col >= 0 && col < cols){
                            sum += input.at(row, col, ch) * kernel[i][j];
                        }
                    }
                }
                output.at(r,c,ch) = sum / divideBy; // Normalize by the sum of the kernel values (for a 3x3 kernel)
            }
        }
    }
}

template void convolve(const Image<uint8_t>& input, Image<uint8_t>& output, const std::vector<std::vector<uint8_t>>& kernel, uint8_t divideBy);