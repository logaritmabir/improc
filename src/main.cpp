#include "ppm_funcs.hpp"
#include <iostream>
#include <exception>
#include <stdexcept>
#include "filters.hpp"
#include <vector>
#include <cstdint>

int main() {
    Image<uint8_t> sample_img;
    try {
        readPPM("../images/sample.ppm", sample_img);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    Image<uint8_t> output_img = sample_img;
    std::vector<std::vector<uint8_t>> kernel = {{1,0,1}, {1,0,1}, {1,0,1}};
    convolve(sample_img, output_img, kernel);

    try {
        savePPM("../images/output.ppm", output_img);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}