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
     * @brief Applies a convolution kernel stored in device global memory.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Pixel type of the convolution kernel (e.g. float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param input Device pointer to the source image
     * @param output Device pointer to the destination image
     * @param kernel Device pointer to a square convolution kernel
     * @param rows Number of image rows
     * @param cols Number of image columns
     * @param kernelWidth Width of the square kernel
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    __global__ void convolution(
        const PixelType* __restrict__ input,
        PixelType* __restrict__ output,
        const KernelType* __restrict__ kernel,
        size_t rows,
        size_t cols,
        size_t kernelWidth
    );

    /**
     * @brief Applies a convolution kernel stored in device global memory and
     *        divides the accumulated result before clamping.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Pixel type of the convolution kernel and divisor (e.g. float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param input Device pointer to the source image
     * @param output Device pointer to the destination image
     * @param kernel Device pointer to a square convolution kernel
     * @param rows Number of image rows
     * @param cols Number of image columns
     * @param kernelWidth Width of the square kernel
     * @param divisor Value used to normalize the convolution sum
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    __global__ void convolution(
        const PixelType* __restrict__ input,
        PixelType* __restrict__ output,
        const KernelType* __restrict__ kernel,
        size_t rows,
        size_t cols,
        size_t kernelWidth,
        KernelType divisor
    );

    /**
     * @brief Convolves a single-channel GPU image with a host kernel pointer.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel type (typically float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param h_kernel Host-side kernel values in row-major order (pointer)
     * @param h_kernelSize Width of the square kernel
     * @throws std::invalid_argument If the images are incompatible or kernel size is invalid
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output, 
        const KernelType* h_kernel, 
        size_t h_kernelSize
    );

    /**
     * @brief Convolves a single-channel GPU image with a host kernel pointer
     *        and divides each convolution sum by @p divisor.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel and divisor type
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param h_kernel Host-side kernel values in row-major order (pointer)
     * @param h_kernelSize Width of the square kernel
     * @param divisor Value used to normalize the convolution sum
     * @throws std::invalid_argument If the images are incompatible, kernel size is invalid,
     *         or @p divisor is zero
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input,
        GpuImage<PixelType>& d_output,
        const KernelType* h_kernel,
        size_t h_kernelSize,
        KernelType divisor
    );

    /**
     * @brief Convolves a single-channel GPU image with a host kernel vector.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel type (typically float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param h_kernel Host-side kernel values in row-major order (vector)
     * @param h_kernelSize Width of the square kernel
     * @throws std::invalid_argument If the vector length does not match
     *         @p h_kernelSize squared or the images are incompatible
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input,
        GpuImage<PixelType>& d_output, 
        const std::vector<KernelType>& h_kernel, 
        size_t h_kernelSize
    );

    /**
     * @brief Convolves a single-channel GPU image with a host kernel vector
     *        and divides each convolution sum by @p divisor.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel and divisor type
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param h_kernel Host-side kernel values in row-major order (vector)
     * @param h_kernelSize Width of the square kernel
     * @param divisor Value used to normalize the convolution sum
     * @throws std::invalid_argument If the vector length does not match
     *         @p h_kernelSize squared, the images are incompatible, or @p divisor is zero
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input,
        GpuImage<PixelType>& d_output,
        const std::vector<KernelType>& h_kernel,
        size_t h_kernelSize,
        KernelType divisor
    );

    /**
     * @brief Applies a median filter to a single-channel GPU image.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param kernelSize Width of the square neighborhood
     * @throws std::invalid_argument If the images are incompatible or the kernel
     *         size is not an odd value in the supported range
     */
    template<typename PixelType>
    void medianFilter(const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output, 
        size_t kernelSize
    );

    /**
     * @brief Applies a Gaussian blur using a kernel derived from sigma.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel type (typically float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     * @param sigma Standard deviation used to build the Gaussian kernel
     * @throws std::invalid_argument If @p sigma is not positive
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void gaussianBlur(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output, 
        float sigma
    );

    /**
     * @brief Applies a default 3x3 Gaussian blur.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel type (typically float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void gaussianBlur(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output
    );

    /**
     * @brief Computes Sobel edge magnitude for a single-channel GPU image.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel type (typically float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void sobelFilter(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output
    );

    /**
     * @brief Applies a 3x3 sharpening filter.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel type (typically float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void sharpeningFilter(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output
    );

    /**
     * @brief Applies a 3x3 Laplacian filter.
     *
     * @tparam PixelType Pixel type of the source and destination images
     * @tparam KernelType Kernel type (typically float)
     * @tparam AccumulatorType Type used for accumulating convolution results
     * @param d_input Source image on the GPU
     * @param d_output Destination image on the GPU
     */
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void laplacianFilter(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output
    );
}
