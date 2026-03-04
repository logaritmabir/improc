#pragma once

#include <cstdint>
#include <vector>

template<typename T>
class Image {
private:
    size_t rows_;
    size_t cols_;
    size_t channels_;
    size_t stride_;
    std::vector<T> data_;
public:
    Image();
    Image(size_t rows, size_t cols, size_t channels);
    Image(const Image& other);
    Image(Image&& other) noexcept;
    ~Image() noexcept;

    size_t rows() const noexcept;
    size_t cols() const noexcept;
    size_t channels() const noexcept;

    T& at(size_t row, size_t col, size_t ch = 0);
    const T& at(size_t row, size_t col, size_t ch = 0) const;

    T* data() noexcept;
    const T* data() const noexcept;

    size_t size() const noexcept;
    size_t stride() const noexcept;

    bool empty() const noexcept;

    typename std::vector<T>::iterator begin();
    typename std::vector<T>::const_iterator begin() const;
    typename std::vector<T>::const_iterator cbegin() const;
    typename std::vector<T>::iterator end();
    typename std::vector<T>::const_iterator end() const;
    typename std::vector<T>::const_iterator cend() const;

    void clear();
    void resize(size_t rows, size_t cols, size_t channels);
    void fill(const T& value);

    Image& operator=(const Image& other);
    Image& operator=(Image&& other) noexcept;
    T& operator()(size_t row, size_t col, size_t ch = 0);
    const T& operator()(size_t row, size_t col, size_t ch = 0) const;
    bool operator==(const Image<T>& other) const;
    bool operator!=(const Image<T>& other) const;
};