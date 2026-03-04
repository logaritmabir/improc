#pragma once
#include "image.hpp"

template<typename T>
void convolve(const Image<T>& input, Image<T>& output, const std::vector<std::vector<T>>& kernel);
