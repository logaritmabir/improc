/**
 * @file gpu_image.cu
 * @brief Implementation of GPU-side image container for CUDA operations.
 *
 * This file contains the template implementations for the GpuImage class,
 * which manages image data on the GPU device. It provides GPU memory
 * allocation, deallocation, and host-device data transfer operations.
 * All CUDA memory operations use the CUDA_CHECK macro for error handling.
 *
 * @see GpuImage for the class interface
 * @note Requires CUDA runtime library for cudaMalloc, cudaMemcpy, and cudaFree
 */

#include "cuda/gpu_image.cuh"
#include "cuda/gpu_utils.cuh"
#include <stdexcept>

namespace cuda {
    
    template<typename PixelType>
    GpuImage<PixelType>::~GpuImage(){
        if (data_) {
            cudaFree(data_);
        }
    }
    template<typename PixelType>
    GpuImage<PixelType>::GpuImage(){}

    template<typename PixelType>
    GpuImage<PixelType>::GpuImage(size_t rows, size_t cols, size_t channels){
        if (rows == 0 || cols == 0 || channels == 0) {
            throw std::invalid_argument("Rows, columns, and channels must be greater than zero.");
        }
        rows_ = rows;
        cols_ = cols;
        channels_ = channels;
        stride_ = cols * channels;
        data_ = nullptr;
        allocate();
    }

    template<typename PixelType>
    GpuImage<PixelType>::GpuImage(const Image<PixelType>& img){
        rows_ = img.rows();
        cols_ = img.cols();
        channels_ = img.channels();
        stride_ = img.stride();
        data_ = nullptr;
        allocate();
    }

    template<typename PixelType>
    size_t GpuImage<PixelType>::cols() const noexcept {
        return cols_;
    }

    template<typename PixelType>
    size_t GpuImage<PixelType>::rows() const noexcept {
        return rows_;
    }

    template<typename PixelType>
    size_t GpuImage<PixelType>::channels() const noexcept {
        return channels_;
    }

    template<typename PixelType>
    size_t GpuImage<PixelType>::stride() const noexcept {
        return stride_;
    }

    template<typename PixelType>
    PixelType* GpuImage<PixelType>::data() noexcept {
        return data_;
    }

    template<typename PixelType>
    const PixelType* GpuImage<PixelType>::data() const noexcept {
        return data_;
    }

    template<typename PixelType>
    void GpuImage<PixelType>::allocate() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
        size_t totalSize = rows_ * stride_ * sizeof(PixelType);
        CUDA_CHECK(cudaMalloc(&data_, totalSize));
    }

    template<typename PixelType>
    void GpuImage<PixelType>::upload(const Image<PixelType>& src) {
        checkSameTypeImages(*this, src);
        if (!data_)
            throw std::runtime_error("GpuImage has no data. Call allocate() first.");

        CUDA_CHECK(cudaMemcpy(data_, src.data(), rows_ * stride_ * sizeof(PixelType), cudaMemcpyHostToDevice));
    }

    template<typename PixelType>
    void GpuImage<PixelType>::download(Image<PixelType>& dst) const {
        checkSameTypeImages(*this, dst);
        if (!data_)
            throw std::runtime_error("GpuImage has no data. Call allocate() and upload() first.");

        size_t totalSize = rows_ * stride_ * sizeof(PixelType);
        CUDA_CHECK(cudaMemcpy(dst.data(), data_, totalSize, cudaMemcpyDeviceToHost));
    }

    template class GpuImage<uint8_t>;
}
