#include "utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

template<typename T>
T clamp(T value, T min, T max) {
    return std::min(std::max(value, min), max);
}

void validateKernelSize(size_t kernel_size) {
    if(kernel_size <= 0)
        throw std::invalid_argument("Kernel size cannot be zero or negative");

    if(kernel_size % 2 == 0)
        throw std::invalid_argument("Kernel size must be odd");
}

template<typename T>
void validateKernel(const std::vector<std::vector<T>>& kernel) {
    if(kernel.empty())
        throw std::invalid_argument("Kernel cannot be empty");
    
    size_t kernelSize = kernel.size();
    
    if(kernelSize % 2 == 0)
        throw std::invalid_argument("Kernel size must be odd");
    
    for(const auto& row : kernel) {
        if(row.size() != kernelSize)
            throw std::invalid_argument("Kernel must be square");
    }
}

template float clamp(float value, float min, float max);
template uint8_t clamp(uint8_t value, uint8_t min, uint8_t max);
template void validateKernel(const std::vector<std::vector<float>>& kernel);
template void validateKernel(const std::vector<std::vector<uint8_t>>& kernel);

template int clamp<int>(int, int, int);