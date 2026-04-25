#include "cuda/morph.cuh"
#include "cuda/gpu_utils.cuh"

#include <stdexcept>

namespace cuda {
    namespace {
        template<typename T>
        __global__ void erosionKernel(const T* input, T* output, size_t rows, size_t cols, size_t stride, size_t kernelSize) {
            const size_t flatIndex = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t total = rows * cols;

            if (flatIndex >= total) {
                return;
            }

            const int row = static_cast<int>(flatIndex / cols);
            const int col = static_cast<int>(flatIndex % cols);
            const int radius = static_cast<int>(kernelSize / 2);
            T minVal = input[index(stride, row, col)];

            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;
                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        const T value = input[index(stride, neighborRow, neighborCol)];
                        if (value < minVal) {
                            minVal = value;
                        }
                    }
                }
            }

            output[index(stride, row, col)] = minVal;
        }

        template<typename T>
        __global__ void dilationKernel(const T* input, T* output, size_t rows, size_t cols, size_t stride, size_t kernelSize) {
            const size_t flatIndex = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t total = rows * cols;

            if (flatIndex >= total) {
                return;
            }

            const int row = static_cast<int>(flatIndex / cols);
            const int col = static_cast<int>(flatIndex % cols);
            const int radius = static_cast<int>(kernelSize / 2);
            T maxVal = input[index(stride, row, col)];

            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;
                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        const T value = input[index(stride, neighborRow, neighborCol)];
                        if (value > maxVal) {
                            maxVal = value;
                        }
                    }
                }
            }

            output[index(stride, row, col)] = maxVal;
        }

        template<typename T>
        __global__ void subtractKernel(const T* left, const T* right, T* output, size_t rows, size_t cols, size_t stride) {
            const size_t flatIndex = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t total = rows * cols;

            if (flatIndex >= total) {
                return;
            }

            const int row = static_cast<int>(flatIndex / cols);
            const int col = static_cast<int>(flatIndex % cols);
            output[index(stride, row, col)] = static_cast<T>(left[index(stride, row, col)] - right[index(stride, row, col)]);
        }

        template<typename T>
        void requireSingleChannelImages(const GpuImage<T>& input, const GpuImage<T>& output) {
            requireSameTypeImages(input, output);
            if (input.channels() != 1 || output.channels() != 1) {
                throw std::invalid_argument("Morphological CUDA operations require single-channel images");
            }
        }

        template<typename T, typename Kernel>
        void launchMorphKernel(const GpuImage<T>& input, GpuImage<T>& output, size_t kernelSize, Kernel kernel) {
            requireSingleChannelImages(input, output);
            requireOddSize(kernelSize);

            const size_t total = input.rows() * input.cols();
            const int threads = 256;
            const int blocks = static_cast<int>((total + threads - 1) / threads);

            kernel<<<blocks, threads>>>(input.data(), output.data(), input.rows(), input.cols(), input.stride(), kernelSize);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        template<typename T>
        void subtractImages(const GpuImage<T>& left, const GpuImage<T>& right, GpuImage<T>& output) {
            requireSingleChannelImages(left, output);
            requireSingleChannelImages(right, output);

            const size_t total = left.rows() * left.cols();
            const int threads = 256;
            const int blocks = static_cast<int>((total + threads - 1) / threads);

            subtractKernel<<<blocks, threads>>>(left.data(), right.data(), output.data(), left.rows(), left.cols(), left.stride());

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    template<typename T>
    void erode(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        launchMorphKernel(d_input, d_output, kernelSize, erosionKernel<T>);
    }

    template<typename T>
    void dilate(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        launchMorphKernel(d_input, d_output, kernelSize, dilationKernel<T>);
    }

    template<typename T>
    void open(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        GpuImage<T> temp(d_input.rows(), d_input.cols(), d_input.channels());
        erode(d_input, temp, kernelSize);
        dilate(temp, d_output, kernelSize);
    }

    template<typename T>
    void close(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        GpuImage<T> temp(d_input.rows(), d_input.cols(), d_input.channels());
        dilate(d_input, temp, kernelSize);
        erode(temp, d_output, kernelSize);
    }

    template<typename T>
    void morphologicalGradient(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        GpuImage<T> dilated(d_input.rows(), d_input.cols(), d_input.channels());
        GpuImage<T> eroded(d_input.rows(), d_input.cols(), d_input.channels());

        dilate(d_input, dilated, kernelSize);
        erode(d_input, eroded, kernelSize);
        subtractImages(dilated, eroded, d_output);
    }

    template<typename T>
    void topHat(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        GpuImage<T> opened(d_input.rows(), d_input.cols(), d_input.channels());
        open(d_input, opened, kernelSize);
        subtractImages(d_input, opened, d_output);
    }

    template<typename T>
    void bottomHat(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        GpuImage<T> closed(d_input.rows(), d_input.cols(), d_input.channels());
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
