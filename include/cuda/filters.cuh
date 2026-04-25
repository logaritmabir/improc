#pragma once

/**
 * @file filters.cuh
 * @brief CUDA image filtering primitives and convenience wrappers.
 *
 * Declares convolution-based CUDA kernels together with host-side wrappers for
 * common spatial filters such as Gaussian blur, Sobel, sharpening, Laplacian,
 * and median filtering on @ref cuda::GpuImage data.
 */

#include "cuda/gpu_image.cuh"
#include <cstddef>
#include <vector>

namespace cuda {

    /**
     * @brief Constant-memory storage for small convolution kernels.
     *
     * Kernels up to 15x15 are copied here to reduce global memory traffic when
     * launching the constant-memory convolution overload.
     */
    __constant__ float c_filter[15 * 15];

    /**
     * @brief Applies a convolution kernel stored in device global memory.
     *
     * @tparam T Pixel type of the source and destination images
     * @param input Device pointer to the source image
     * @param output Device pointer to the destination image
     * @param kernel Device pointer to a square convolution kernel
     * @param rows Number of image rows
     * @param cols Number of image columns
     * @param kernelWidth Width of the square kernel
     */
    template<typename T>
    __global__ void convolution(
        const T* __restrict__ input,
        T* __restrict__ output,
        const float* __restrict__ kernel,
        size_t rows,
        size_t cols,
        size_t kernelWidth
    );

    /**
     * @brief Applies a convolution kernel stored in constant memory.
     *
     * @tparam T Pixel type of the source and destination images
     * @param input Device pointer to the source image
     * @param output Device pointer to the destination image
     * @param rows Number of image rows
     * @param cols Number of image columns
     * @param kernelWidth Width of the square kernel already copied to
     *        @ref c_filter
     */
    template<typename T>
    __global__ void convolution(
        const T* __restrict__ input,
        T* __restrict__ output,
        size_t rows,
        size_t cols,
        size_t kernelWidth
    );

    /**
     * @brief Convolves a single-channel GPU image with a host kernel buffer.
     *
     * Small kernels are copied to constant memory automatically; larger kernels
     * are uploaded to device memory for the launch.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param h_kernel Host pointer to a square convolution kernel
     * @param h_kernelSize Width of the square kernel
     * @throws std::invalid_argument If the images are incompatible or the kernel
     *         size is not odd
     */
    template<typename T>
    void convolve(const GpuImage<T>& d_input, GpuImage<T>& d_output, const float* h_kernel, size_t h_kernelSize);

    /**
     * @brief Convolves a single-channel GPU image with a host kernel vector.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param h_kernel Host-side kernel values in row-major order
     * @param h_kernelSize Width of the square kernel
     * @throws std::invalid_argument If the vector length does not match
     *         @p h_kernelSize squared or the images are incompatible
     */
    template<typename T>
    void convolve(const GpuImage<T>& d_input, GpuImage<T>& d_output, const std::vector<float>& h_kernel, size_t h_kernelSize);

    /**
     * @brief Applies a median filter to a single-channel GPU image.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square neighborhood
     * @throws std::invalid_argument If the images are incompatible or the kernel
     *         size is not an odd value in the supported range
     */
    template<typename T>
    void medianFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize);

    /**
     * @brief Applies a Gaussian blur using a kernel derived from sigma.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param sigma Standard deviation used to build the Gaussian kernel
     * @throws std::invalid_argument If @p sigma is not positive
     */
    template<typename T>
    void gaussianBlur(const GpuImage<T>& d_input, GpuImage<T>& d_output, float sigma);

    /**
     * @brief Applies a default 3x3 Gaussian blur.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename T>
    void gaussianBlur(const GpuImage<T>& d_input, GpuImage<T>& d_output);

    /**
     * @brief Computes Sobel edge magnitude for a single-channel GPU image.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename T>
    void sobelFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output);

    /**
     * @brief Applies a 3x3 sharpening filter.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename T>
    void sharpeningFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output);

    /**
     * @brief Applies a 3x3 Laplacian filter.
     *
     * @tparam T Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename T>
    void laplacianFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output);
}
