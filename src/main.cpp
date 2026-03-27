#include <iostream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <cstdint>

#include "pnm_funcs.hpp"
#include "filters.hpp"
#include "cuda/gpu_image.cuh"
#include "cuda/kernels.cuh"

int main() {
    Image<uint8_t> sample_img;
    try {
        readPNM("../images/sampleGRAY.pgm", sample_img);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Convolution test
    Image<uint8_t> convolution_img = sample_img;
    std::vector<std::vector<float>> kernel = {{0.0625f, 0.1250f, 0.0625f}, {0.1250f, 0.2500f, 0.1250f}, {0.0625f, 0.1250f, 0.0625f}};
    convolve(sample_img, convolution_img, kernel);
 
    // Median filter test
    Image<uint8_t> medianFilter_img = sample_img;
    medianFilter(sample_img, medianFilter_img, 3);

    // Sobel filter test
    Image<uint8_t> sobelFilter_img = sample_img;
    std::vector<std::vector<float>> sobelKernelX = {{-1.0f, 0.0f, 1.0f}, {-2.0f, 0.0f, 2.0f}, {-1.0f, 0.0f, 1.0f}};
    std::vector<std::vector<float>> sobelKernelY = {{-1.0f, -2.0f, -1.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}};

    convolve(sample_img, sobelFilter_img, sobelKernelX);
    convolve(sample_img, sobelFilter_img, sobelKernelY);

    // Invert test
    Image<uint8_t> inverted_img = sample_img;
    invert(sample_img, inverted_img);

    // Threshold test
    Image<uint8_t> threshold_img = sample_img;
    threshold(sample_img, threshold_img, (uint8_t)128);

    // CUDA convolution test
    Image<uint8_t> h_convolution_img = sample_img;
    cuda::GpuImage<uint8_t> d_input(h_convolution_img); //image is allocated
    cuda::GpuImage<uint8_t> d_output(h_convolution_img);

    d_input.upload(h_convolution_img); //data is uploaded to gpu

    const float h_kernel[] = {
    0.0625f, 0.1250f, 0.0625f,
    0.1250f, 0.2500f, 0.1250f,
    0.0625f, 0.1250f, 0.0625f};

    cuda::convolve(d_input, d_output, h_kernel, 3);

    d_output.download(h_convolution_img);

    try {
        savePNM("../images/convolution_img.pgm", convolution_img);
        savePNM("../images/medianFilter_img.pgm", medianFilter_img);
        savePNM("../images/sobelFilter_img.pgm", sobelFilter_img);
        savePNM("../images/inverted_img.pgm", inverted_img);
        savePNM("../images/threshold_img.pgm", threshold_img);
        savePNM("../images/h_convolution_img.pgm", h_convolution_img);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
