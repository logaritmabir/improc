#include "morph.hpp"
#include "utils.hpp"
#include <limits>

template<typename T>
void erode(const Image<T>& input, Image<T>& output, size_t kernel_size){
    validateKernelSize(kernel_size);

    if(input.rows() != output.rows() || input.cols() != output.cols())
        throw std::invalid_argument("Input and output images must have the same dimensions");

    int filterRadius = static_cast<int>(kernel_size / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            T minVal = std::numeric_limits<T>::max();

            for(int i = -filterRadius; i <= filterRadius; ++i) {
                for(int j = -filterRadius; j <= filterRadius; ++j) {
                    int neighborRow = r + i;
                    int neighborCol = c + j;

                    if(neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        T pixelVal = input(neighborRow, neighborCol);
                        minVal = std::min(minVal, pixelVal);
                    }
                }
            }

            output(r, c) = minVal;
        }
    }
}

template<typename T>
void dilate(const Image<T>& input, Image<T>& output, size_t kernel_size){
    validateKernelSize(kernel_size);

    if(input.rows() != output.rows() || input.cols() != output.cols())
        throw std::invalid_argument("Input and output images must have the same dimensions");

    int filterRadius = static_cast<int>(kernel_size / 2);
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());

    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            T maxVal = std::numeric_limits<T>::min();

            for(int i = -filterRadius; i <= filterRadius; ++i) {
                for(int j = -filterRadius; j <= filterRadius; ++j) {
                    int neighborRow = r + i;
                    int neighborCol = c + j;

                    if(neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        T pixelVal = input(neighborRow, neighborCol);
                        maxVal = std::max(maxVal, pixelVal);
                    }
                }
            }

            output(r, c) = maxVal;
        }
    }
}

template void erode(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
template void dilate(const Image<uint8_t>& input, Image<uint8_t>& output, size_t kernel_size);
    
