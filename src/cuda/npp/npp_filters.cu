#include "cuda/npp/npp_filters.cuh"

#include <stdexcept>
#include <string>

// relocate npp namespace under cuda namespace MAYBE!!!!!
namespace npp {
    void convolution(
        const cuda::GpuImage<uint8_t>& d_input, 
        cuda::GpuImage<uint8_t>& d_output, 
        const float* h_kernel, 
        size_t h_kernelSize){
            cuda::requireSingleChannelImage(d_input);
            cuda::requireSameTypeImages(d_input, d_output);
            cuda::requireOddSize(h_kernelSize);
            
            float* d_kernel = nullptr;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kernel), h_kernelSize * h_kernelSize * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, h_kernelSize * h_kernelSize * sizeof(float), cudaMemcpyHostToDevice));

            const NppStatus status = nppiFilterBorder32f_8u_C1R_Ctx(
                d_input.data(), 
                d_input.stride() * sizeof(uint8_t),
                {static_cast<int>(d_input.cols()), static_cast<int>(d_input.rows())},
                {0, 0},  
                d_output.data(), 
                d_output.stride() * sizeof(uint8_t),
                {static_cast<int>(d_input.cols()), static_cast<int>(d_input.rows())}, 
                d_kernel, 
                {static_cast<int>(h_kernelSize), static_cast<int>(h_kernelSize)}, 
                {static_cast<int>(h_kernelSize / 2), static_cast<int>(h_kernelSize / 2)},
                NPP_BORDER_REPLICATE,
                {}
            );

            CUDA_CHECK(cudaFree(d_kernel));

            if (status != NPP_SUCCESS) {
                throw std::runtime_error("nppiFilterBorder32f_8u_C1R failed with status " + std::to_string(status));
            }

            CUDA_CHECK(cudaDeviceSynchronize());
        }
} // namespace npp
