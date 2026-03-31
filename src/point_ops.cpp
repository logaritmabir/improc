#include "point_ops.hpp"
#include <stdexcept>

template<typename T>
void invert(const Image<T>& input, Image<T>& output) {
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");
    
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            for(int ch = 0; ch < channels; ch++){
                output(r,c,ch) = static_cast<T>(255 - input(r,c,ch));
            }
        }
    }
}

template<typename T>
void threshold(const Image<T>& input, Image<T>& output, T thresholdValue) {
    if(input.rows() != output.rows() || input.cols() != output.cols() || input.channels() != output.channels())
        throw std::invalid_argument("Input and output images must have the same dimensions and number of channels");
    
    int cols = static_cast<int>(input.cols());
    int rows = static_cast<int>(input.rows());
    int channels = static_cast<int>(input.channels());

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            for(int ch = 0; ch < channels; ch++){
                output(r,c,ch) = (input(r,c,ch) >= thresholdValue) ? static_cast<T>(255) : static_cast<T>(0);
            }
        }
    }
}

template void invert(const Image<uint8_t>& input, Image<uint8_t>& output);
template void threshold(const Image<uint8_t>& input, Image<uint8_t>& output, uint8_t thresholdValue);
