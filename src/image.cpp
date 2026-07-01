/**
 * @file image.cpp
 * @brief Implementation of the Image class template for multi-channel image data.
 *
 * This file contains the template implementations for the Image class,
 * which provides a flexible container for image data with support for
 * multiple data types and channels. All member functions are defined here
 * as template instantiations.
 *
 * @see Image for the class interface
 */

#include "image.hpp"
#include <stdexcept>
#include <algorithm>

template<typename PixelType>
Image<PixelType>::Image() : rows_(0), cols_(0), channels_(0), stride_(0), data_() {}

template<typename PixelType>
Image<PixelType>::Image(size_t rows, size_t cols, size_t channels) {
        if (rows == 0 || cols == 0 || channels == 0) {
            throw std::invalid_argument("Rows, columns, and channels must be greater than zero.");
        }
        rows_ = rows;
        cols_ = cols;
        channels_ = channels;
        stride_ = cols * channels;
        data_.resize(rows * cols * channels);
    }

template<typename PixelType>
Image<PixelType>::Image(const Image<PixelType>& other)
    : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_), stride_(other.stride_), data_(other.data_) {}

template<typename PixelType>
Image<PixelType>::Image(Image<PixelType>&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), channels_(other.channels_), stride_(other.stride_), data_(std::move(other.data_)) {
    other.cols_ = 0;
    other.rows_ = 0;
    other.channels_ = 0;
    other.stride_ = 0;
}

template<typename PixelType>
Image<PixelType>::~Image() noexcept {}

template<typename PixelType>
size_t Image<PixelType>::rows() const noexcept {
    return rows_;
}

template<typename PixelType>
size_t Image<PixelType>::cols() const noexcept {
    return cols_;
}

template<typename PixelType>
size_t Image<PixelType>::channels() const noexcept {
    return channels_;
}

template<typename PixelType>
PixelType& Image<PixelType>::at(size_t row, size_t col, size_t ch) {
    if (row >= rows_)
        throw std::out_of_range("Row index out of range");

    if (col >= cols_)
        throw std::out_of_range("Column index out of range");

    if (ch >= channels_)
        throw std::out_of_range("Channel index out of range");

    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename PixelType>
const PixelType& Image<PixelType>::at(size_t row, size_t col, size_t ch) const{
    if (row >= rows_)
        throw std::out_of_range("Row index out of range");

    if (col >= cols_)
        throw std::out_of_range("Column index out of range");

    if (ch >= channels_)
        throw std::out_of_range("Channel index out of range");

    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename PixelType>
PixelType* Image<PixelType>::data() noexcept {
    return data_.data();
}

template<typename PixelType>
const PixelType* Image<PixelType>::data() const noexcept {
    return data_.data();
}

template<typename PixelType>
size_t Image<PixelType>::size() const noexcept {
    return data_.size();
}

template<typename PixelType>
size_t Image<PixelType>::stride() const noexcept {
    return stride_;
}

template<typename PixelType>
bool Image<PixelType>::empty() const noexcept {
    return data_.empty();
}

template<typename PixelType>
typename std::vector<PixelType>::iterator Image<PixelType>::begin(){
    return data_.begin();
}

template<typename PixelType>
typename std::vector<PixelType>::const_iterator Image<PixelType>::begin() const{
    return data_.begin();
}

template<typename PixelType>
typename std::vector<PixelType>::const_iterator Image<PixelType>::cbegin() const{
    return data_.cbegin();
}

template<typename PixelType>
typename std::vector<PixelType>::iterator Image<PixelType>::end(){
    return data_.end();
}

template<typename PixelType>
typename std::vector<PixelType>::const_iterator Image<PixelType>::end() const{
    return data_.end();
}

template<typename PixelType>
typename std::vector<PixelType>::const_iterator Image<PixelType>::cend() const{
    return data_.cend();
}

template<typename PixelType>
void Image<PixelType>::fill(const PixelType& value){
    std::fill(data_.begin(),data_.end(), value);
}

template<typename PixelType>
void Image<PixelType>::clear(){
    rows_ = 0;
    cols_ = 0;
    channels_ = 0;
    stride_ = 0;
    data_.clear();
}

template<typename PixelType>
void Image<PixelType>::resize(size_t rows, size_t cols, size_t channels) {
    if (rows == 0 || cols == 0 || channels == 0)
        throw std::invalid_argument("Dimensions must be > 0");
    rows_ = rows;
    cols_ = cols;
    channels_ = channels;
    stride_ = cols * channels;
    data_.resize(rows * cols * channels);
}


template<typename PixelType>
Image<PixelType>& Image<PixelType>::operator=(const Image<PixelType>& other) {
    if (this != &other) {
        cols_ = other.cols_;
        rows_ = other.rows_;
        channels_ = other.channels_;
        stride_ = other.stride_;
        data_ = other.data_;
    }
    return *this;
}

template<typename PixelType>
Image<PixelType>& Image<PixelType>::operator=(Image<PixelType>&& other) noexcept {
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

template<typename PixelType>
PixelType& Image<PixelType>::operator()(size_t row, size_t col, size_t ch) {
    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename PixelType>
const PixelType& Image<PixelType>::operator()(size_t row, size_t col, size_t ch) const{
    return data_[(row * stride_) + col * channels_ + ch];
}

template<typename PixelType>
bool Image<PixelType>::operator==(const Image<PixelType>& other) const {
    return (rows_ == other.rows_ && cols_ == other.cols_ && channels_ == other.channels_ && stride_ == other.stride_ && data_ == other.data_);
}

template<typename PixelType>
bool Image<PixelType>::operator!=(const Image<PixelType>& other) const {
    return !(*this == other);
}

template class Image<uint8_t>;
template class Image<int>;
