/**
 * @file morph.cu
 * @brief Implementation of GPU-accelerated morphological image operations.
 *
 * This file contains CUDA kernel implementations for morphological operations
 * including erosion, dilation, opening, closing, morphological gradient,
 * and top/bottom-hat transformations. All operations are optimized for
 * GPU execution on single-channel images.
 *
 * @see morph.cuh for the function declarations and detailed documentation
 * @note All morphological operations check single-channel (grayscale) GPU images
 */

#include "cuda/morph.cuh"
#include "cuda/gpu_utils.cuh"

#include <stdexcept>

namespace cuda {
    namespace {
        template<typename PixelType>
        __global__ void erosionKernel(const PixelType* input, PixelType* output, size_t rows, size_t cols, size_t stride, size_t kernelSize) {
            const size_t flatIndex = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t total = rows * cols;

            if (flatIndex >= total) {
                return;
            }

            const int row = static_cast<int>(flatIndex / cols);
            const int col = static_cast<int>(flatIndex % cols);
            const int radius = static_cast<int>(kernelSize / 2);
            PixelType minVal = input[index(stride, row, col)];

            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;
                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        const PixelType value = input[index(stride, neighborRow, neighborCol)];
                        if (value < minVal) {
                            minVal = value;
                        }
                    }
                }
            }

            output[index(stride, row, col)] = minVal;
        }

        template<typename PixelType>
        __global__ void dilationKernel(const PixelType* input, PixelType* output, size_t rows, size_t cols, size_t stride, size_t kernelSize) {
            const size_t flatIndex = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t total = rows * cols;

            if (flatIndex >= total) {
                return;
            }

            const int row = static_cast<int>(flatIndex / cols);
            const int col = static_cast<int>(flatIndex % cols);
            const int radius = static_cast<int>(kernelSize / 2);
            PixelType maxVal = input[index(stride, row, col)];

            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;
                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        const PixelType value = input[index(stride, neighborRow, neighborCol)];
                        if (value > maxVal) {
                            maxVal = value;
                        }
                    }
                }
            }

            output[index(stride, row, col)] = maxVal;
        }

        template<typename PixelType>
        __global__ void subtractKernel(const PixelType* left, const PixelType* right, PixelType* output, size_t rows, size_t cols, size_t stride) {
            const size_t flatIndex = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t total = rows * cols;

            if (flatIndex >= total) {
                return;
            }

            const int row = static_cast<int>(flatIndex / cols);
            const int col = static_cast<int>(flatIndex % cols);
            output[index(stride, row, col)] = static_cast<PixelType>(left[index(stride, row, col)] - right[index(stride, row, col)]);
        }

        template<typename PixelType>
        void checkSingleChannelImages(const GpuImage<PixelType>& input, const GpuImage<PixelType>& output) {
            checkSameTypeImages(input, output);
            if (input.channels() != 1 || output.channels() != 1) {
                throw std::invalid_argument("Morphological CUDA operations check single-channel images");
            }
        }

        template<typename PixelType, typename KernelFunction>
        void launchMorphKernel(const GpuImage<PixelType>& input, GpuImage<PixelType>& output, size_t kernelSize, KernelFunction kernel) {
            checkSingleChannelImages(input, output);
            checkOddSize(kernelSize);

            const size_t total = input.rows() * input.cols();
            const int threads = 256;
            const int blocks = static_cast<int>((total + threads - 1) / threads);

            kernel<<<blocks, threads>>>(input.data(), output.data(), input.rows(), input.cols(), input.stride(), kernelSize);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        template<typename PixelType>
        void subtractImages(const GpuImage<PixelType>& left, const GpuImage<PixelType>& right, GpuImage<PixelType>& output) {
            checkSingleChannelImages(left, output);
            checkSingleChannelImages(right, output);

            const size_t total = left.rows() * left.cols();
            const int threads = 256;
            const int blocks = static_cast<int>((total + threads - 1) / threads);

            subtractKernel<<<blocks, threads>>>(left.data(), right.data(), output.data(), left.rows(), left.cols(), left.stride());

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    template<typename PixelType>
    void erode(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        launchMorphKernel(d_input, d_output, kernelSize, erosionKernel<PixelType>);
    }

    template<typename PixelType>
    void dilate(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        launchMorphKernel(d_input, d_output, kernelSize, dilationKernel<PixelType>);
    }

    template<typename PixelType>
    void open(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        GpuImage<PixelType> temp(d_input.rows(), d_input.cols(), d_input.channels());
        erode(d_input, temp, kernelSize);
        dilate(temp, d_output, kernelSize);
    }

    template<typename PixelType>
    void close(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        GpuImage<PixelType> temp(d_input.rows(), d_input.cols(), d_input.channels());
        dilate(d_input, temp, kernelSize);
        erode(temp, d_output, kernelSize);
    }

    template<typename PixelType>
    void morphologicalGradient(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        GpuImage<PixelType> dilated(d_input.rows(), d_input.cols(), d_input.channels());
        GpuImage<PixelType> eroded(d_input.rows(), d_input.cols(), d_input.channels());

        dilate(d_input, dilated, kernelSize);
        erode(d_input, eroded, kernelSize);
        subtractImages(dilated, eroded, d_output);
    }

    template<typename PixelType>
    void topHat(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        GpuImage<PixelType> opened(d_input.rows(), d_input.cols(), d_input.channels());
        open(d_input, opened, kernelSize);
        subtractImages(d_input, opened, d_output);
    }

    template<typename PixelType>
    void bottomHat(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        GpuImage<PixelType> closed(d_input.rows(), d_input.cols(), d_input.channels());
        close(d_input, closed, kernelSize);
        subtractImages(closed, d_input, d_output);
    }

    template void erode(const GpuImage<uint8_t>& d_input, GpuImage<uint8_t>& d_output, size_t kernelSize);
    template void dilate(const GpuImage<uint8_t>& d_input, GpuImage<uint8_t>& d_output, size_t kernelSize);
    template void open(const GpuImage<uint8_t>& d_input, GpuImage<uint8_t>& d_output, size_t kernelSize);
    template void close(const GpuImage<uint8_t>& d_input, GpuImage<uint8_t>& d_output, size_t kernelSize);
    template void morphologicalGradient(const GpuImage<uint8_t>& d_input, GpuImage<uint8_t>& d_output, size_t kernelSize);
    template void topHat(const GpuImage<uint8_t>& d_input, GpuImage<uint8_t>& d_output, size_t kernelSize);
    template void bottomHat(const GpuImage<uint8_t>& d_input, GpuImage<uint8_t>& d_output, size_t kernelSize);
}
