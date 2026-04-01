/**
 * @mainpage Improc Image Processing Library
 *
 * @brief A C++ image processing library with CPU and GPU (CUDA) support.
 *
 * @section overview Overview
 *
 * Improc provides a comprehensive set of image processing operations including:
 * - Image container class with multi-channel support
 * - Point operations (inversion, thresholding)
 * - Filtering operations (convolution, median filter, Gaussian blur, Sobel edge detection)
 * - PNM file format I/O (PGM for grayscale, PPM for RGB)
 * - GPU-accelerated convolution via CUDA
 *
 * @section usage Usage
 *
 * @subsection basic Basic Example
 * @code
 * #include "pnm_funcs.hpp"
 * #include "filters.hpp"
 *
 * Image<uint8_t> img;
 * readPNM("input.pgm", img);
 *
 * Image<uint8_t> blurred;
 * gaussianBlur(img, blurred, 2.0f);
 *
 * savePNM("output.pgm", blurred);
 * @endcode
 *
 * @subsection cuda CUDA Example
 * @code
 * #include "cuda/gpu_image.cuh"
 * #include "cuda/kernels.cuh"
 *
 * cuda::GpuImage<uint8_t> d_input(img);
 * cuda::GpuImage<uint8_t> d_output(img);
 *
 * d_input.upload(img);
 *
 * float kernel[] = { 0.0625f, 0.1250f, 0.0625f,
 *                    0.1250f, 0.2500f, 0.1250f,
 *                    0.0625f, 0.1250f, 0.0625f };
 *
 * cuda::convolve(d_input, d_output, kernel, 3);
 * d_output.download(result);
 * @endcode
 *
 * @section modules Modules
 *
 * - @ref Image "Image Class" - Template-based image container
 * - @ref point_ops "Point Operations" - Per-pixel transformations
 * - @ref filters "Filtering" - Convolution and smoothing filters
 * - @ref pnm_funcs "PNM I/O" - File format support
 * - @ref cuda "CUDA Support" - GPU-accelerated processing
 *
 * @section build Building
 *
 * The project uses a Makefile for building:
 * @code
 * make        # Build the project
 * make clean  # Clean build artifacts
 * make run    # Build and run the example
 * @endcode
 *
 * @section license License
 *
 * This project is provided as-is for educational purposes.
 */

#include <iostream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <cstdint>

#include "pnm_funcs.hpp"
#include "filters.hpp"
#ifdef BUILD_WITH_CUDA
#include "cuda/gpu_image.cuh"
#include "cuda/kernels.cuh"
#endif
#include "point_ops.hpp"

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

    #ifdef BUILD_WITH_CUDA
    // CUDA convolution test
    Image<uint8_t> h_convolution_img = sample_img;
    cuda::GpuImage<uint8_t> d_input(h_convolution_img);
    cuda::GpuImage<uint8_t> d_output(h_convolution_img);

    d_input.upload(h_convolution_img);

    const float h_kernel[] = {
    0.0625f, 0.1250f, 0.0625f,
    0.1250f, 0.2500f, 0.1250f,
    0.0625f, 0.1250f, 0.0625f};

    cuda::convolve(d_input, d_output, h_kernel, 3);

    d_output.download(h_convolution_img);
    #endif

    try {
        savePNM("../images/convolution_img.pgm", convolution_img);
        savePNM("../images/medianFilter_img.pgm", medianFilter_img);
        savePNM("../images/sobelFilter_img.pgm", sobelFilter_img);
        savePNM("../images/inverted_img.pgm", inverted_img);
        savePNM("../images/threshold_img.pgm", threshold_img);
        #ifdef BUILD_WITH_CUDA
        savePNM("../images/h_convolution_img.pgm", h_convolution_img);
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
