#include "image.hpp"
#include <stdexcept>
#include <algorithm>

template<typename T>
Image<T>::Image() : rows_(0), cols_(0), channels_(0), stride_(0), data_() {}

template<typename T>
Image<T>::Image(size_t rows, size_t cols, size_t channels) {
        if (rows == 0 || cols == 0 || channels == 0) {
            throw std::invalid_argument("Rows, columns, and channels must be greater than zero.");
        }
        rows_ = rows;
        cols_ = cols;
        channels_ = channels;
        stride_ = cols * channels;
        data_.resize(rows * cols * channels);
    }

template<typename T>
Image<T>::Image(const Image<T>& other)
    : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_), stride_(other.stride_), data_(other.data_) {}

template<typename T>
Image<T>::Image(Image<T>&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_), stride_(other.stride_), data_(std::move(other.data_)) {
    other.cols_ = 0;
    other.rows_ = 0;
    other.channels_ = 0;
    other.stride_ = 0;
}

template<typename T>
Image<T>::~Image() noexcept {}

template<typename T>
size_t Image<T>::rows() const noexcept {
    return rows_;
}

template<typename T>
size_t Image<T>::cols() const noexcept {
    return cols_;
}

template<typename T>
size_t Image<T>::channels() const noexcept {
    return channels_;
}

template<typename T>
T& Image<T>::at(size_t row, size_t col, size_t ch) {
    if (row >= rows_)
        throw std::out_of_range("Row index out of range");

    if (col >= cols_)
        throw std::out_of_range("Column index out of range");

    if (ch >= channels_)
        throw std::out_of_range("Channel index out of range");

    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename T>
const T& Image<T>::at(size_t row, size_t col, size_t ch) const{
    if (row >= rows_)
        throw std::out_of_range("Row index out of range");

    if (col >= cols_)
        throw std::out_of_range("Column index out of range");

    if (ch >= channels_)
        throw std::out_of_range("Channel index out of range");

    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename T>
T* Image<T>::data() noexcept {
    return data_.data();
}

template<typename T>
const T* Image<T>::data() const noexcept {
    return data_.data();
}

template<typename T>
size_t Image<T>::size() const noexcept {
    return data_.size();
}

template<typename T>
size_t Image<T>::stride() const noexcept {
    return stride_;
}

template<typename T>
bool Image<T>::empty() const noexcept {
    return data_.empty();
}

template<typename T>
typename std::vector<T>::iterator Image<T>::begin(){
    return data_.begin();
}

template<typename T>
typename std::vector<T>::const_iterator Image<T>::begin() const{
    return data_.begin();
}

template<typename T>
typename std::vector<T>::const_iterator Image<T>::cbegin() const{
    return data_.cbegin();
}

template<typename T>
typename std::vector<T>::iterator Image<T>::end(){
    return data_.end();
}

template<typename T>
typename std::vector<T>::const_iterator Image<T>::end() const{
    return data_.end();
}

template<typename T>
typename std::vector<T>::const_iterator Image<T>::cend() const{
    return data_.cend();
}

template<typename T>
void Image<T>::fill(const T& value){
    std::fill(data_.begin(),data_.end(), value);
}

template<typename T>
void Image<T>::clear(){
    rows_ = 0;
    cols_ = 0;
    channels_ = 0;
    stride_ = 0;
    data_.clear();
}

template<typename T>
void Image<T>::resize(size_t rows, size_t cols, size_t channels) {
    if (rows == 0 || cols == 0 || channels == 0)
        throw std::invalid_argument("Dimensions must be > 0");
    rows_ = rows;
    cols_ = cols;
    channels_ = channels;
    stride_ = cols * channels;
    data_.resize(rows * cols * channels);
}


template<typename T>
Image<T>& Image<T>::operator=(const Image<T>& other) {
    if (this != &other) {
        cols_ = other.cols_;
        rows_ = other.rows_;
        channels_ = other.channels_;
        stride_ = other.stride_;
        data_ = other.data_;
    }
    return *this;
}

template<typename T>
Image<T>& Image<T>::operator=(Image<T>&& other) noexcept {
    if (this != &other) {
        cols_ = other.cols_;
        rows_ = other.rows_;
        channels_ = other.channels_;
        stride_ = other.stride_;
        data_ = std::move(other.data_);
        other.cols_ = 0;
        other.rows_ = 0;
        other.channels_ = 0;
        other.stride_ = 0;
    }
    return *this;
}

template<typename T>
T& Image<T>::operator()(size_t row, size_t col, size_t ch) {
    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename T>
const T& Image<T>::operator()(size_t row, size_t col, size_t ch) const{
    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename T>
bool Image<T>::operator==(const Image<T>& other) const {
    return (rows_ == other.rows_ && cols_ == other.cols_ && channels_ == other.channels_ && stride_ == other.stride_ && data_ == other.data_);
}

template<typename T>
bool Image<T>::operator!=(const Image<T>& other) const {
    return !(*this == other);
}

template class Image<uint8_t>;
template class Image<int>;
