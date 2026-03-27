#include "cuda/kernels.cuh"

namespace cuda {

    __device__ __forceinline__ size_t index(size_t cols, size_t x, size_t y){
        return (x * cols + y);
    }

    template<typename T>
    __global__ void convolution(const T* input, T* output, const float* kernel, size_t rows, size_t cols, size_t kernelWidth) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col < cols && row < rows) {
            float sum = 0.0f;
            int k = kernelWidth / 2;

            for(int i = -k; i <= k; i++){
                for(int j = -k; j <= k; j++){
                    int posX = row + i;
                    int posY = col + j;

                    if(posX < rows && posX >= 0 && posY < cols && posY >= 0){
                        sum += input[index(cols, posX, posY)] * kernel[index(kernelWidth, i + k, j + k)];
                    }
                }
            }
            output[index(cols, row, col)] = static_cast<T>(sum);
        }
    }

    template<typename T>
    void convolve(const GpuImage<T>& d_input, GpuImage<T>& d_output, const float* h_kernel, size_t h_kernelSize) {

        if(d_input.rows() != d_output.rows() || d_input.cols() != d_output.cols()){
            throw std::runtime_error("Input and output images must have the same dimensions.");
        }
        if(h_kernelSize % 2 == 0){
            throw std::runtime_error("Kernel size must be odd.");
        }

        dim3 threads(16, 16);
        dim3 blocks((d_input.cols() + threads.x - 1) / threads.x, (d_input.rows() + threads.y - 1) / threads.y);

        const float* d_kernel;
        CUDA_CHECK(cudaMalloc((void**)&d_kernel, h_kernelSize * h_kernelSize * sizeof(float)));
        CUDA_CHECK(cudaMemcpy((void*)d_kernel, (const void*)h_kernel, h_kernelSize * h_kernelSize * sizeof(float), cudaMemcpyHostToDevice));

        convolution<<<blocks, threads>>>(d_input.data(), d_output.data(), d_kernel, d_input.rows(), d_input.cols(), h_kernelSize);

        CUDA_CHECK(cudaFree((void*)d_kernel));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template void convolve(const GpuImage<uint8_t>& input, GpuImage<uint8_t>& output, const float* kernel, size_t kernelSize);
}
