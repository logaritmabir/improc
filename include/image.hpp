#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @class Image
 * @brief A template-based container for storing multi-channel image data.
 *
 * The Image class provides a flexible container for storing image data with support
 * for multiple channels (grayscale, RGB, etc.) and type flexibility. Data is stored
 * in row-major order with optional stride support.
 *
 * This class is designed to work with images loaded from PNM format files (PGM for
 * grayscale, PPM for RGB color). See pnm_funcs.hpp for PNM I/O operations.
 *
 * @tparam T The data type for pixel values (e.g., uint8_t, float)
 */
template<typename T>
class Image {
private:
    size_t rows_;
    size_t cols_;
    size_t channels_;
    size_t stride_;
    std::vector<T> data_;
public:
    /// @name Constructors and Destructors
    /// @{

    /**
     * @brief Default constructor.
     *
     * Creates an empty image with zero dimensions.
     */
    Image();

    /**
     * @brief Constructs an image with specified dimensions.
     *
     * @param rows Number of rows in the image
     * @param cols Number of columns in the image
     * @param channels Number of channels per pixel
     */
    Image(size_t rows, size_t cols, size_t channels);

    /**
     * @brief Copy constructor.
     *
     * @param other The image to copy from
     */
    Image(const Image& other);

    /**
     * @brief Move constructor.
     *
     * @param other The image to move from
     */
    Image(Image&& other) noexcept;

    /**
     * @brief Destructor.
     */
    ~Image() noexcept;

    /// @}

    /// @name Accessors
    /// @{

    /**
     * @brief Returns the number of rows in the image.
     *
     * @return Number of rows
     */
    size_t rows() const noexcept;

    /**
     * @brief Returns the number of columns in the image.
     *
     * @return Number of columns
     */
    size_t cols() const noexcept;

    /**
     * @brief Returns the number of channels per pixel.
     *
     * @return Number of channels (1 for grayscale, 3 for RGB, etc.)
     */
    size_t channels() const noexcept;

    /**
     * @brief Accesses a pixel value at the specified position.
     *
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @param ch Channel index (default: 0)
     * @return Reference to the pixel value
     */
    T& at(size_t row, size_t col, size_t ch = 0);

    /**
     * @brief Accesses a pixel value at the specified position (const version).
     *
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @param ch Channel index (default: 0)
     * @return Const reference to the pixel value
     */
    const T& at(size_t row, size_t col, size_t ch = 0) const;

    /**
     * @brief Returns a pointer to the raw image data.
     *
     * @return Pointer to the image data buffer
     */
    T* data() noexcept;

    /**
     * @brief Returns a const pointer to the raw image data.
     *
     * @return Const pointer to the image data buffer
     */
    const T* data() const noexcept;

    /**
     * @brief Returns the total number of elements in the image.
     *
     * @return Total size (rows × cols × channels)
     */
    size_t size() const noexcept;

    /**
     * @brief Returns the stride (number of elements per row including all channels).
     *
     * @return Stride value
     */
    size_t stride() const noexcept;

    /**
     * @brief Checks if the image is empty.
     *
     * @return true if the image has zero dimensions, false otherwise
     */
    bool empty() const noexcept;

    /// @}

    /// @name Iterators
    /// @{

    /**
     * @brief Returns an iterator to the beginning of the image data.
     *
     * @return Iterator to the first element
     */
    typename std::vector<T>::iterator begin();

    /**
     * @brief Returns a const iterator to the beginning of the image data.
     *
     * @return Const iterator to the first element
     */
    typename std::vector<T>::const_iterator begin() const;

    /**
     * @brief Returns a const iterator to the beginning of the image data.
     *
     * @return Const iterator to the first element
     */
    typename std::vector<T>::const_iterator cbegin() const;

    /**
     * @brief Returns an iterator to the end of the image data.
     *
     * @return Iterator to one past the last element
     */
    typename std::vector<T>::iterator end();

    /**
     * @brief Returns a const iterator to the end of the image data.
     *
     * @return Const iterator to one past the last element
     */
    typename std::vector<T>::const_iterator end() const;

    /**
     * @brief Returns a const iterator to the end of the image data.
     *
     * @return Const iterator to one past the last element
     */
    typename std::vector<T>::const_iterator cend() const;

    /// @}

    /// @}

    /// @name Modifiers
    /// @{

    /**
     * @brief Clears the image, freeing all data.
     */
    void clear();

    /**
     * @brief Resizes the image to new dimensions.
     *
     * @param rows New number of rows
     * @param cols New number of columns
     * @param channels New number of channels
     */
    void resize(size_t rows, size_t cols, size_t channels);

    /**
     * @brief Fills the entire image with a constant value.
     *
     * @param value The value to fill with
     */
    void fill(const T& value);

    /// @}

    /// @name Assignment Operators
    /// @{

    /**
     * @brief Copy assignment operator.
     *
     * @param other The image to assign from
     * @return Reference to this image
     */
    Image& operator=(const Image& other);

    /**
     * @brief Move assignment operator.
     *
     * @param other The image to assign from
     * @return Reference to this image
     */
    Image& operator=(Image&& other) noexcept;

    /// @}

    /// @name Access Operators
    /// @{

    /**
     * @brief Accesses a pixel value using function call syntax.
     *
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @param ch Channel index (default: 0)
     * @return Reference to the pixel value
     */
    T& operator()(size_t row, size_t col, size_t ch = 0);

    /**
     * @brief Accesses a pixel value using function call syntax (const version).
     *
     * @param row Row index (0-based)
     * @param col Column index (0-based)
     * @param ch Channel index (default: 0)
     * @return Const reference to the pixel value
     */
    const T& operator()(size_t row, size_t col, size_t ch = 0) const;

    /// @}

    /// @name Comparison Operators
    /// @{

    /**
     * @brief Equality comparison operator.
     *
     * @param other The image to compare with
     * @return true if images are equal, false otherwise
     */
    bool operator==(const Image<T>& other) const;

    /**
     * @brief Inequality comparison operator.
     *
     * @param other The image to compare with
     * @return true if images are not equal, false otherwise
     */
    bool operator!=(const Image<T>& other) const;

    /// @}
};