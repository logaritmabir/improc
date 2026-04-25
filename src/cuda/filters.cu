#include "cuda/filters.cuh"
#include "cuda/gpu_utils.cuh"

#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cuda {
    template<typename T>
    __global__ void convolution(
        const T* __restrict__ input,
        T* __restrict__ output,
        const float* __restrict__ kernel,
        size_t rows,
        size_t cols,
        size_t kernelWidth
    ) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int filterRadius = static_cast<int>(kernelWidth / 2);

        if (col < cols && row < rows) {
            float sum = 0.0f;

            for (int i = -filterRadius; i <= filterRadius; ++i) {
                for (int j = -filterRadius; j <= filterRadius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;

                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        sum += static_cast<float>(input[index(cols, neighborRow, neighborCol)]) *
                            kernel[index(kernelWidth, i + filterRadius, j + filterRadius)];
                    }
                }
            }

            output[index(cols, row, col)] = static_cast<T>(clamp<float>(sum, 0.0f, 255.0f));
        }
    }

    template<typename T>
    __global__ void convolution(
        const T* __restrict__ input,
        T* __restrict__ output,
        size_t rows,
        size_t cols,
        size_t kernelWidth
    ) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int filterRadius = static_cast<int>(kernelWidth / 2);

        if (col < cols && row < rows) {
            float sum = 0.0f;

            for (int i = -filterRadius; i <= filterRadius; ++i) {
                for (int j = -filterRadius; j <= filterRadius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;

                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        sum += static_cast<float>(input[index(cols, neighborRow, neighborCol)]) *
                            c_filter[index(kernelWidth, i + filterRadius, j + filterRadius)];
                    }
                }
            }

            output[index(cols, row, col)] = static_cast<T>(clamp<float>(sum, 0.0f, 255.0f));
        }
    }

    template<typename T>
    __global__ void medianFilterKernel(
        const T* input,
        T* output,
        size_t rows,
        size_t cols,
        size_t kernelWidth
    ) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int filterRadius = static_cast<int>(kernelWidth / 2);

        if (col < cols && row < rows) {
            unsigned char neighbors[225];
            int count = 0;

            for (int i = -filterRadius; i <= filterRadius; ++i) {
                for (int j = -filterRadius; j <= filterRadius; ++j) {
                    const int neighborRow = row + i;
                    const int neighborCol = col + j;

                    if (neighborRow >= 0 && neighborRow < rows && neighborCol >= 0 && neighborCol < cols) {
                        neighbors[count++] = static_cast<unsigned char>(input[index(cols, neighborRow, neighborCol)]);
                    }
                }
            }

            for (int i = 1; i < count; ++i) {
                const unsigned char value = neighbors[i];
                int j = i - 1;
                while (j >= 0 && neighbors[j] > value) {
                    neighbors[j + 1] = neighbors[j];
                    --j;
                }
                neighbors[j + 1] = value;
            }

            output[index(cols, row, col)] = static_cast<T>(neighbors[count / 2]);
        }
    }

    template<typename T>
    void convolve(const GpuImage<T>& d_input, GpuImage<T>& d_output, const float* h_kernel, size_t h_kernelSize) {
        requireSingleChannelImage(d_input);
        requireSameTypeImages(d_input, d_output);
        requireOddSize(h_kernelSize);

        dim3 threads(16, 16);
        dim3 blocks((d_input.cols() + threads.x - 1) / threads.x,(d_input.rows() + threads.y - 1) / threads.y);

        if(h_kernelSize <= 15) {
            CUDA_CHECK(cudaMemcpyToSymbol(c_filter, h_kernel, h_kernelSize * h_kernelSize * sizeof(float)));
            convolution<<<blocks, threads>>>(
                d_input.data(),
                d_output.data(),
                d_input.rows(),
                d_input.cols(),
                h_kernelSize
            );
        } else {
            float* d_kernel;
            CUDA_CHECK(cudaMalloc(&d_kernel, h_kernelSize * h_kernelSize * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, h_kernelSize * h_kernelSize * sizeof(float), cudaMemcpyHostToDevice));
            convolution<<<blocks, threads>>>(
                d_input.data(),
                d_output.data(),
                d_kernel,
                d_input.rows(),
                d_input.cols(),
                h_kernelSize
            );
            CUDA_CHECK(cudaFree(d_kernel));
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template<typename T>
    void convolve(const GpuImage<T>& d_input, GpuImage<T>& d_output, const std::vector<float>& h_kernel, size_t h_kernelSize) {
        if (h_kernel.size() != h_kernelSize * h_kernelSize) {
            throw std::invalid_argument("Kernel size does not match kernel data");
        }
        convolve(d_input, d_output, h_kernel.data(), h_kernelSize);
    }

    template<typename T>
    void medianFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output, size_t kernelSize) {
        requireSingleChannelImage(d_input);
        requireSameTypeImages(d_input, d_output);
        requireOddSize(kernelSize);

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

    template<typename T>
    void gaussianBlur(const GpuImage<T>& d_input, GpuImage<T>& d_output, float sigma) {
        if (sigma <= 0.0f) {
            throw std::invalid_argument("Sigma must be greater than 0");
        }

        const size_t kernelSize = static_cast<size_t>(static_cast<int>(std::ceil(sigma * 6.0f)) | 1);
        const int filterRadius = static_cast<int>(kernelSize / 2);
        std::vector<float> kernel(kernelSize * kernelSize);
        float sum = 0.0f;

        for (int i = -filterRadius; i <= filterRadius; ++i) {
            for (int j = -filterRadius; j <= filterRadius; ++j) {
                const float exponent = -(i * i + j * j) / (2.0f * sigma * sigma);
                const float value = std::exp(exponent);
                kernel[static_cast<size_t>(i + filterRadius) * kernelSize + static_cast<size_t>(j + filterRadius)] = value;
                sum += value;
            }
        }

        for (float& value : kernel) {
            value /= sum;
        }

        convolve(d_input, d_output, kernel, kernelSize);
    }

    template<typename T>
    void gaussianBlur(const GpuImage<T>& d_input, GpuImage<T>& d_output) {
        static const std::array<float, 9> kernel = {
            0.0625f, 0.1250f, 0.0625f,
            0.1250f, 0.2500f, 0.1250f,
            0.0625f, 0.1250f, 0.0625f
        };
        convolve(d_input, d_output, kernel.data(), 3);
    }

    template<typename T>
    void sobelFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output) {
        static_assert(std::is_same<T, uint8_t>::value, "CUDA sobelFilter is currently instantiated for uint8_t only");
        requireSameTypeImages(d_input, d_output);
        requireSingleChannelImage(d_input);

        static const std::array<float, 9> gx = {
            -1.0f, 0.0f, 1.0f,
            -2.0f, 0.0f, 2.0f,
            -1.0f, 0.0f, 1.0f
        };
        static const std::array<float, 9> gy = {
            1.0f, 2.0f, 1.0f,
            0.0f, 0.0f, 0.0f,
            -1.0f, -2.0f, -1.0f
        };

        GpuImage<uint8_t> d_gradX(d_input.rows(), d_input.cols(), d_input.channels());
        GpuImage<uint8_t> d_gradY(d_input.rows(), d_input.cols(), d_input.channels());

        convolve(d_input, d_gradX, gx.data(), 3);
        convolve(d_input, d_gradY, gy.data(), 3);

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

    template<typename T>
    void sharpeningFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output) {
        static const std::array<float, 9> kernel = {
            0.0f, -1.0f, 0.0f,
            -1.0f, 5.0f, -1.0f,
            0.0f, -1.0f, 0.0f
        };
        convolve(d_input, d_output, kernel.data(), 3);
    }

    template<typename T>
    void laplacianFilter(const GpuImage<T>& d_input, GpuImage<T>& d_output) {
        static const std::array<float, 9> kernel = {
            0.0f, 1.0f, 0.0f,
            1.0f, -4.0f, 1.0f,
            0.0f, 1.0f, 0.0f
        };
        convolve(d_input, d_output, kernel.data(), 3);
    }

    template void convolve(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const float* kernel, size_t kernelSize);
    template void convolve(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const std::vector<float>& kernel, size_t kernelSize);
    template void medianFilter(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, size_t kernelSize);
    template void gaussianBlur(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, float sigma);
    template void gaussianBlur(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
    template void sobelFilter(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
    template void sharpeningFilter(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
    template void laplacianFilter(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output);
}
