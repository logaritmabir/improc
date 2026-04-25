#include "cuda/gpu_image.cuh"
#include "cuda/gpu_utils.cuh"
#include <stdexcept>

namespace cuda {
    
    template<typename T>
    GpuImage<T>::~GpuImage(){
        if (data_) {
            cudaFree(data_);
        }
    }
    template<typename T>
    GpuImage<T>::GpuImage(){}

    template<typename T>
    GpuImage<T>::GpuImage(size_t rows, size_t cols, size_t channels){
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

    template<typename T>
    GpuImage<T>::GpuImage(const Image<T>& img){
        rows_ = img.rows();
        cols_ = img.cols();
        channels_ = img.channels();
        stride_ = img.stride();
        data_ = nullptr;
        allocate();
    }

    template<typename T>
    size_t GpuImage<T>::cols() const noexcept {
        return cols_;
    }

    template<typename T>
    size_t GpuImage<T>::rows() const noexcept {
        return rows_;
    }

    template<typename T>
    size_t GpuImage<T>::channels() const noexcept {
        return channels_;
    }

    template<typename T>
    size_t GpuImage<T>::stride() const noexcept {
        return stride_;
    }

    template<typename T>
    T* GpuImage<T>::data() noexcept {
        return data_;
    }

    template<typename T>
    const T* GpuImage<T>::data() const noexcept {
        return data_;
    }

    template<typename T>
    void GpuImage<T>::allocate() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
        size_t totalSize = rows_ * stride_ * sizeof(T);
        CUDA_CHECK(cudaMalloc(&data_, totalSize));
    }

    template<typename T>
    void GpuImage<T>::upload(const Image<T>& src) {
        requireSameTypeImages(*this, src);
        if (!data_)
            throw std::runtime_error("GpuImage has no data. Call allocate() first.");

        CUDA_CHECK(cudaMemcpy(data_, src.data(), rows_ * stride_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void GpuImage<T>::download(Image<T>& dst) const {
        requireSameTypeImages(*this, dst);
        if (!data_)
            throw std::runtime_error("GpuImage has no data. Call allocate() and upload() first.");

        size_t totalSize = rows_ * stride_ * sizeof(T);
        CUDA_CHECK(cudaMemcpy(dst.data(), data_, totalSize, cudaMemcpyDeviceToHost));
    }

    template class GpuImage<uint8_t>;
}
