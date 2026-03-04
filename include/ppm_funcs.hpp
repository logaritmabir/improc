#pragma once

#include "image.hpp"
#include <string>

template<typename T>
void readPPM(const std::string& filename, Image<T>& img);

template<typename T>
void savePPM(const std::string& filename, Image<T>& img);
