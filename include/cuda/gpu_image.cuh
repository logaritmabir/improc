#pragma once

/**
 * @file gpu_image.cuh
 * @brief GPU-side image container and memory management for CUDA operations.
 *
 * This module provides the GpuImage class for managing image data on the GPU,
 * including memory allocation, host-to-device and device-to-host data transfer.
 *
 * @note This is a CUDA header file and requires NVCC or a compatible compiler.
 * @see Image for the corresponding CPU-side image container
 */

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "image.hpp"

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error( \
            std::string(cudaGetErrorString(err)) + \
            " at line: " + std::to_string(__LINE__) + " at file : " + std::string(__FILE__) \
        ); \
    } \
} while (0)

namespace cuda {

    /**
     * @class GpuImage
     * @brief GPU-side image container for CUDA image processing.
     *
     * The GpuImage class mirrors the CPU-side Image class but stores data
     * in GPU memory. It provides methods for allocating GPU memory and
     * transferring data between host (CPU) and device (GPU).
     *
     * @tparam PixelType The pixel data type; the library currently explicitly
     * instantiates this class for uint8_t.
     */
    template<typename PixelType>
    class GpuImage{
    private:
        size_t rows_;
        size_t cols_;
        size_t channels_;
        size_t stride_;
        PixelType* data_;

    public:
        /// @name Constructors
        /// @{

        /**
         * @brief Default constructor.
         *
         * Creates an empty GpuImage with zero dimensions and null data pointer.
         */
        GpuImage();

        /**
         * @brief Constructs a GpuImage with specified dimensions.
         *
         * Allocates GPU memory immediately upon construction.
         *
         * @param rows Number of rows in the image
         * @param cols Number of columns in the image
         * @param channels Number of channels per pixel
         * @throws std::invalid_argument if any dimension is zero
         * @throws std::runtime_error If CUDA memory allocation fails
         */
        GpuImage(size_t rows, size_t cols, size_t channels);

        /**
         * @brief Constructs a GpuImage from a CPU Image.
         *
         * Copies dimensions from the source Image and allocates GPU memory.
         * Does NOT copy pixel data - use upload() to transfer data.
         *
         * @param img Source CPU-side Image to copy dimensions from
         * @throws std::runtime_error If CUDA memory allocation fails
         */
        explicit GpuImage(const Image<PixelType>& img);

        /**
         * @brief Copy construction is disabled because this class owns GPU memory.
         */
        GpuImage(const GpuImage& other) = delete;

        /**
         * @brief Transfers GPU memory ownership from another image.
         * @param other Image to move from; it becomes empty.
         */
        GpuImage(GpuImage&& other) noexcept;

        /// @}

        /// @name Destructor
        /// @{

        /**
         * @brief Destructor.
         *
         * Frees GPU memory if data was allocated.
         */
        ~GpuImage();

        /// @}

        /// @name Assignment Operators
        /// @{

        /**
         * @brief Copy assignment is disabled because this class owns GPU memory.
         */
        GpuImage& operator=(const GpuImage& other) = delete;

        /**
         * @brief Transfers GPU memory ownership from another image.
         * @param other Image to move from; it becomes empty.
         * @return Reference to this image.
         */
        GpuImage& operator=(GpuImage&& other) noexcept;

        /// @}

        /// @name Dimension Accessors
        /// @{

        /**
         * @brief Returns the number of columns.
         * @return Number of columns
         */
        size_t cols() const noexcept;

        /**
         * @brief Returns the number of rows.
         * @return Number of rows
         */
        size_t rows() const noexcept;

        /**
         * @brief Returns the number of channels.
         * @return Number of channels
         */
        size_t channels() const noexcept;

        /**
         * @brief Returns the stride (elements per row including channels).
         * @return Stride value
         */
        size_t stride() const noexcept;

        /// @}

        /// @name Data Accessors
        /// @{

        /**
         * @brief Returns pointer to GPU data.
         * @return Pointer to the GPU data buffer
         */
        PixelType* data() noexcept;

        /**
         * @brief Returns const pointer to GPU data.
         * @return Const pointer to the GPU data buffer
         */
        const PixelType* data() const noexcept;

        /// @}

        /// @name Memory Management
        /// @{

        /**
         * @brief Allocates GPU memory for the image.
         *
         * If memory was previously allocated, it is freed first. The image
         * must have nonzero dimensions.
         *
         * @throws std::runtime_error If CUDA memory allocation fails
         */
        void allocate();

        /**
         * @brief Uploads data from CPU Image to GPU memory.
         *
         * @param src Source CPU Image to upload
         * @throws std::invalid_argument If image metadata does not match
         * @throws std::runtime_error If GPU memory is not allocated or a CUDA
         * transfer fails
         */
        void upload(const Image<PixelType>& src);

        /**
         * @brief Downloads data from GPU memory to CPU Image.
         *
         * @param dst Destination CPU Image to download to
         * @throws std::invalid_argument If image metadata does not match
         * @throws std::runtime_error If GPU memory is not allocated or a CUDA
         * transfer fails
         */
        void download(Image<PixelType>& dst) const;
        /// @}

    };
}
