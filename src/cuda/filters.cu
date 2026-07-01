/**
 * @file filters.cu
 * @brief Implementation of CUDA-accelerated image filtering operations.
 *
 * This file contains CUDA kernel implementations and host-side wrappers for
 * various spatial filters including convolution, Gaussian blur, Sobel edge
 * detection, sharpening, Laplacian, and median filtering. Operations are
 * optimized for GPU execution with support for both global and constant memory.
 *
 * @see filters.cuh for the function declarations and detailed documentation
 * @note Uses NVIDIA Performance Primitives (NPP) for optimized implementations where available
 */

#include "cuda/filters.cuh"
#include "cuda/gpu_utils.cuh"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

//todo accumulators
namespace cuda {
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    __global__ void convolution(
        const PixelType* __restrict__ input,
        PixelType* __restrict__ output,
        const KernelType* __restrict__ kernel,
        size_t rows,
        size_t cols,
        size_t kernelWidth
    ) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int filterRadius = static_cast<int>(kernelWidth / 2);

        if (col < cols && row < rows) {
            AccumulatorType sum = AccumulatorType{0};

            for (int i = -filterRadius; i <= filterRadius; ++i) {
                for (int j = -filterRadius; j <= filterRadius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;

                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        sum += static_cast<AccumulatorType>(input[index(cols, neighborRow, neighborCol)]) *
                            static_cast<AccumulatorType>(kernel[index(kernelWidth, i + filterRadius, j + filterRadius)]);
                    }
                }
            }

            output[index(cols, row, col)] = static_cast<PixelType>(clamp<AccumulatorType>(sum, AccumulatorType{0}, AccumulatorType{255}));
        }
    }

    // Accumulator type will be type generic in later. At least should be
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    __global__ void convolution(
        const PixelType* __restrict__ input,
        PixelType* __restrict__ output,
        const KernelType* __restrict__ kernel,
        size_t rows,
        size_t cols,
        size_t kernelWidth,
        KernelType divisor
    ) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int filterRadius = static_cast<int>(kernelWidth / 2);

        if (col < cols && row < rows) {
            AccumulatorType sum = AccumulatorType{0};

            for (int i = -filterRadius; i <= filterRadius; ++i) {
                for (int j = -filterRadius; j <= filterRadius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;

                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        sum += static_cast<AccumulatorType>(input[index(cols, neighborRow, neighborCol)]) *
                            static_cast<AccumulatorType>(kernel[index(kernelWidth, i + filterRadius, j + filterRadius)]);
                    }
                }
            }
            sum /= static_cast<AccumulatorType>(divisor);
            output[index(cols, row, col)] = static_cast<PixelType>(clamp<AccumulatorType>(sum, AccumulatorType{0}, AccumulatorType{255}));
        }
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output, 
        const KernelType* h_kernel, 
        size_t h_kernelSize
    ) {
        checkSingleChannelImage(d_input);
        checkSameTypeImages(d_input, d_output);
        checkOddSize(h_kernelSize);

        dim3 threads(16, 16);
        dim3 blocks((d_input.cols() + threads.x - 1) / threads.x,(d_input.rows() + threads.y - 1) / threads.y);

        KernelType* d_kernel;
        CUDA_CHECK(cudaMalloc(&d_kernel, h_kernelSize * h_kernelSize * sizeof(KernelType)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, h_kernelSize * h_kernelSize * sizeof(KernelType), cudaMemcpyHostToDevice));
        convolution<PixelType,KernelType,AccumulatorType><<<blocks, threads>>>(
            d_input.data(),
            d_output.data(),
            d_kernel,
            d_input.rows(),
            d_input.cols(),
            h_kernelSize
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input,
        GpuImage<PixelType>& d_output,
        const KernelType* h_kernel,
        size_t h_kernelSize,
        KernelType divisor
    ) {
        checkSingleChannelImage(d_input);
        checkSameTypeImages(d_input, d_output);
        checkOddSize(h_kernelSize);

        if (divisor == KernelType{0}) {
            throw std::invalid_argument("Divisor must not be zero");
        }

        dim3 threads(16, 16);
        dim3 blocks((d_input.cols() + threads.x - 1) / threads.x, (d_input.rows() + threads.y - 1) / threads.y);

        KernelType* d_kernel;
        CUDA_CHECK(cudaMalloc(&d_kernel, h_kernelSize * h_kernelSize * sizeof(KernelType)));
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, h_kernelSize * h_kernelSize * sizeof(KernelType), cudaMemcpyHostToDevice));
        convolution<PixelType,KernelType,AccumulatorType><<<blocks, threads>>>(
            d_input.data(),
            d_output.data(),
            d_kernel,
            d_input.rows(),
            d_input.cols(),
            h_kernelSize,
            divisor
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_kernel));
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input, 
        GpuImage<PixelType>& d_output, 
        const std::vector<KernelType>& h_kernel, 
        size_t h_kernelSize
    ) {
        if (h_kernel.size() != h_kernelSize * h_kernelSize) {
            throw std::invalid_argument("Kernel size does not match kernel data");
        }
        convolve<PixelType,KernelType,AccumulatorType>(d_input, d_output, h_kernel.data(), h_kernelSize);
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void convolve(
        const GpuImage<PixelType>& d_input,
        GpuImage<PixelType>& d_output,
        const std::vector<KernelType>& h_kernel,
        size_t h_kernelSize,
        KernelType divisor
    ) {
        if (h_kernel.size() != h_kernelSize * h_kernelSize) {
            throw std::invalid_argument("Kernel size does not match kernel data");
        }
        convolve<PixelType,KernelType,AccumulatorType>(d_input, d_output, h_kernel.data(), h_kernelSize, divisor);
    }

    template<typename PixelType>
    __global__ void medianFilterKernel(const PixelType* input, PixelType* output, size_t rows, size_t cols, size_t kernelWidth) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int filterRadius = static_cast<int>(kernelWidth / 2);

        if (col < cols && row < rows) {
            PixelType neighbors[225];
            int count = 0;

            for (int i = -filterRadius; i <= filterRadius; ++i) {
                for (int j = -filterRadius; j <= filterRadius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;

                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        neighbors[count++] = static_cast<PixelType>(input[index(cols, neighborRow, neighborCol)]);
                    }
                }
            }

            for (int i = 1; i < count; ++i) {
                const PixelType value = neighbors[i];
                int j = i - 1;
                while (j >= 0 && neighbors[j] > value) {
                    neighbors[j + 1] = neighbors[j];
                    --j;
                }
                neighbors[j + 1] = value;
            }

            output[index(cols, row, col)] = static_cast<PixelType>(neighbors[count / 2]);
        }
    }

    template<typename PixelType>
    void medianFilter(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, size_t kernelSize) {
        checkSingleChannelImage(d_input);
        checkSameTypeImages(d_input, d_output);
        checkOddSize(kernelSize);

        if (kernelSize <= 1) {
            throw std::invalid_argument("Kernel size must be greater than 1 and odd");
        }
        if (kernelSize * kernelSize > 225) {
            throw std::invalid_argument("CUDA median filter currently supports kernels up to 15x15");
        }

        dim3 threads(16, 16);
        dim3 blocks((d_input.cols() + threads.x - 1) / threads.x, (d_input.rows() + threads.y - 1) / threads.y);

        medianFilterKernel<<<blocks, threads>>>(
            d_input.data(),
            d_output.data(),
            d_input.rows(),
            d_input.cols(),
            kernelSize
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void gaussianBlur(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output, float sigma) {
        if (sigma <= 0.0f) {
            throw std::invalid_argument("Sigma must be greater than 0");
        }

        const size_t kernelSize = static_cast<size_t>(static_cast<int>(std::ceil(sigma * 6.0f)) | 1);
        const int filterRadius = static_cast<int>(kernelSize / 2);
        std::vector<KernelType> kernel(kernelSize * kernelSize);
        KernelType sum = KernelType{0};

        for (int i = -filterRadius; i <= filterRadius; ++i) {
            for (int j = -filterRadius; j <= filterRadius; ++j) {
                const float exponent = -(i * i + j * j) / (2.0f * sigma * sigma);
                const float value = std::exp(exponent);
                kernel[static_cast<size_t>(i + filterRadius) * kernelSize + static_cast<size_t>(j + filterRadius)] = static_cast<KernelType>(value);
                sum += static_cast<KernelType>(value);
            }
        }

        for (KernelType& value : kernel) {
            value /= sum;
        }

        convolve<PixelType, KernelType, AccumulatorType>(d_input, d_output, kernel, kernelSize);
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void gaussianBlur(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output) {
        static const std::array<KernelType, 9> kernel = {
            KernelType{0.0625}, KernelType{0.1250}, KernelType{0.0625},
            KernelType{0.1250}, KernelType{0.2500}, KernelType{0.1250},
            KernelType{0.0625}, KernelType{0.1250}, KernelType{0.0625}
        };
        convolve<PixelType, KernelType, AccumulatorType>(d_input, d_output, kernel.data(), 3);
    }
    
    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void sobelFilter(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output) {
        static_assert(std::is_same<PixelType, uint8_t>::value, "CUDA sobelFilter is currently instantiated for uint8_t only");
        checkSameTypeImages(d_input, d_output);
        checkSingleChannelImage(d_input);

        static const std::array<KernelType, 9> gx = {
            KernelType{-1.0}, KernelType{0.0}, KernelType{1.0},
            KernelType{-2.0}, KernelType{0.0}, KernelType{2.0},
            KernelType{-1.0}, KernelType{0.0}, KernelType{1.0}
        };
        static const std::array<KernelType, 9> gy = {
            KernelType{1.0}, KernelType{2.0}, KernelType{1.0},
            KernelType{0.0}, KernelType{0.0}, KernelType{0.0},
            KernelType{-1.0}, KernelType{-2.0}, KernelType{-1.0}
        };

        GpuImage<uint8_t> d_gradX(d_input.rows(), d_input.cols(), d_input.channels());
        GpuImage<uint8_t> d_gradY(d_input.rows(), d_input.cols(), d_input.channels());

        convolve<PixelType, KernelType, AccumulatorType>(d_input, d_gradX, gx.data(), 3);
        convolve<PixelType, KernelType, AccumulatorType>(d_input, d_gradY, gy.data(), 3);

        Image<uint8_t> h_gradX(d_input.rows(), d_input.cols(), d_input.channels());
        Image<uint8_t> h_gradY(d_input.rows(), d_input.cols(), d_input.channels());
        Image<uint8_t> h_output(d_output.rows(), d_output.cols(), d_output.channels());

        d_gradX.download(h_gradX);
        d_gradY.download(h_gradY);

        for (size_t row = 0; row < h_output.rows(); ++row) {
            for (size_t col = 0; col < h_output.cols(); ++col) {
                const float gradX = static_cast<float>(h_gradX(row, col));
                const float gradY = static_cast<float>(h_gradY(row, col));
                const float magnitude = std::sqrt(gradX * gradX + gradY * gradY);
                h_output(row, col) = static_cast<uint8_t>(std::min(std::max(magnitude, 0.0f), 255.0f));
            }
        }

        d_output.upload(h_output);
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void sharpeningFilter(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output) {
        static const std::array<KernelType, 9> kernel = {
            KernelType{0.0}, KernelType{-1.0}, KernelType{0.0},
            KernelType{-1.0}, KernelType{5.0}, KernelType{-1.0},
            KernelType{0.0}, KernelType{-1.0}, KernelType{0.0}
        };
        convolve<PixelType, KernelType, AccumulatorType>(d_input, d_output, kernel.data(), 3);
    }

    template<typename PixelType, typename KernelType, typename AccumulatorType>
    void laplacianFilter(const GpuImage<PixelType>& d_input, GpuImage<PixelType>& d_output) {
        static const std::array<KernelType, 9> kernel = {
            KernelType{0.0}, KernelType{1.0}, KernelType{0.0},
            KernelType{1.0}, KernelType{-4.0}, KernelType{1.0},
            KernelType{0.0}, KernelType{1.0}, KernelType{0.0}
        };
        convolve<PixelType, KernelType, AccumulatorType>(d_input, d_output, kernel.data(), 3);
    }



    template void convolve<uint8_t, float, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const float* kernel, size_t kernelSize);
    template void convolve<uint8_t, uint8_t, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const uint8_t* kernel, size_t kernelSize, uint8_t divisor);
    template void convolve<uint8_t, uint8_t, uint32_t>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const uint8_t* kernel, size_t kernelSize, uint8_t divisor);
    template void convolve<uint8_t, float, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const std::vector<float>& kernel, size_t kernelSize);
    template void convolve<uint8_t, uint8_t, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const std::vector<uint8_t>& kernel, size_t kernelSize, uint8_t divisor);
    template void medianFilter(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, size_t kernelSize);
    template void gaussianBlur<uint8_t, float, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, float sigma);
    template void gaussianBlur<uint8_t, float, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
    template void sobelFilter<uint8_t, float, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
    template void sharpeningFilter<uint8_t, float, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
    template void laplacianFilter<uint8_t, float, float>(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
}
