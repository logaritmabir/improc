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
 * - Filtering operations (convolution, median filter, Gaussian blur, Sobel edge detection, sharpening, Laplacian)
 * - Morphological operations (erosion, dilation)
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
 * - @ref morph "Morphological Operations" - Erosion and dilation
 * - @ref pnm_funcs "PNM I/O" - File format support
 * - @ref cuda "CUDA Support" - GPU-accelerated processing
 *
 * @section build Building
 *
 * The project uses a Makefile for building:
 * @code
 * make        # Build the project
 * make -DBUILD_WITH_CUDA=1 # Build the project with CUDA support
 * make clean  # Clean build artifacts
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
#include "cuda/filters.cuh"
#include "cuda/morph.cuh"
#include "cuda/npp/npp_filters.cuh"
#endif
#include "point_ops.hpp"
#include "morph.hpp"

int main() {
    Image<uint8_t> sample_img;
    try {
        readPNM("../images/samples/sampleGRAY.pgm", sample_img);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Gaussian blur test
    Image<uint8_t> gaussianBlur_img = sample_img;
    Image<uint8_t> gaussianBlur_without_sigma_img = sample_img;
    gaussianBlur(sample_img, gaussianBlur_without_sigma_img);
    gaussianBlur(sample_img, gaussianBlur_img, 0.3);

    // Median filter test
    Image<uint8_t> medianFilter_img = sample_img;
    medianFilter(sample_img, medianFilter_img, 3);

    // Sobel filter test
    Image<uint8_t> sobelFilter_img = sample_img;
    sobelFilter(sample_img, sobelFilter_img);

    // Invert test
    Image<uint8_t> inverted_img = sample_img;
    invert(sample_img, inverted_img);

    // Threshold test
    Image<uint8_t> threshold_img = sample_img;
    threshold(sample_img, threshold_img, (uint8_t)128);

    // Sharpening test
    Image<uint8_t> sharpening_img = sample_img;
    sharpeningFilter(sample_img, sharpening_img);

    // Laplacian filter test
    Image<uint8_t> laplacian_img = sample_img;
    laplacianFilter(sample_img, laplacian_img);

    // Morphological operations test
    Image<uint8_t> eroded_img = threshold_img;
    erode(sample_img, eroded_img, 3);

    Image<uint8_t> dilated_img = threshold_img;
    dilate(sample_img, dilated_img, 3);

    Image<uint8_t> opened_img = threshold_img;
    open(sample_img, opened_img, 3);

    Image<uint8_t> closed_img = threshold_img;
    close(sample_img, closed_img, 3);

    Image<uint8_t> morphGradient_img = threshold_img;
    morphologicalGradient(sample_img, morphGradient_img, 3);

    Image<uint8_t> topHat_img = sample_img;
    topHat(sample_img, topHat_img, 3);

    Image<uint8_t> bottomHat_img = sample_img;
    bottomHat(sample_img, bottomHat_img, 3);

    // Histogram equalization test
    Image<uint8_t> histogramEqualized_img = sample_img;
    histogramEqualization(sample_img, histogramEqualized_img);

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

    // NPP convolution test
    Image<uint8_t> h_npp_convolution_img = sample_img;
    cuda::GpuImage<uint8_t> d_npp_input(h_npp_convolution_img);
    cuda::GpuImage<uint8_t> d_npp_output(h_npp_convolution_img);
    d_npp_input.upload(h_npp_convolution_img);

    npp::convolution(d_npp_input, d_npp_output, h_kernel, 3);
    d_npp_output.download(h_npp_convolution_img);

    #endif

    try {
        savePNM("../images/outputs/gaussianBlur_img.pgm", gaussianBlur_img);
        savePNM("../images/outputs/gaussianBlur_without_sigma_img.pgm", gaussianBlur_without_sigma_img);
        savePNM("../images/outputs/medianFilter_img.pgm", medianFilter_img);
        savePNM("../images/outputs/sobelFilter_img.pgm", sobelFilter_img);
        savePNM("../images/outputs/inverted_img.pgm", inverted_img);
        savePNM("../images/outputs/threshold_img.pgm", threshold_img);
        savePNM("../images/outputs/sharpening_img.pgm", sharpening_img);
        savePNM("../images/outputs/laplacian_img.pgm", laplacian_img);
        savePNM("../images/outputs/eroded_img.pgm", eroded_img);
        savePNM("../images/outputs/dilated_img.pgm", dilated_img);
        savePNM("../images/outputs/opened_img.pgm", opened_img);
        savePNM("../images/outputs/closed_img.pgm", closed_img);
        savePNM("../images/outputs/morphGradient_img.pgm", morphGradient_img);
        savePNM("../images/outputs/topHat_img.pgm", topHat_img);
        savePNM("../images/outputs/bottomHat_img.pgm", bottomHat_img);
        savePNM("../images/outputs/histogramEqualized_img.pgm", histogramEqualized_img);
        #ifdef BUILD_WITH_CUDA
        savePNM("../images/outputs/h_convolution_img.pgm", h_convolution_img);
        savePNM("../images/outputs/h_npp_convolution_img.pgm", h_npp_convolution_img);
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
