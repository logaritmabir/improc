#include "pnm_funcs.hpp"
#include <iostream>
#include <exception>
#include <stdexcept>
#include "filters.hpp"
#include <vector>
#include <cstdint>

int main() {
    Image<uint8_t> sample_img;
    try {
        readPNM("../images/sampleGRAY.pgm", sample_img);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    Image<uint8_t> output_img = sample_img;
    std::vector<std::vector<uint8_t>> kernel = {{1,0,1}, {1,0,1}, {1,0,1}};
    uint8_t divideBy = 6; // Sum of the kernel values for normalization
    convolve(sample_img, output_img, kernel, divideBy);

    try {
        savePNM("../images/output.pgm", output_img);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}