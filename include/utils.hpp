#pragma once

/**
 * @file utils.hpp
 * @brief Utility functions for image processing operations.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "image.hpp"

/**
 * @brief Clamps a value to a specified range.
 *
 * @tparam T The data type for the value (must support comparison operators)
 * @param value The value to clamp
 * @param min The minimum bound (inclusive)
 * @param max The maximum bound (inclusive)
 * @return The clamped value within [min, max]
 */
template<typename T>
T clamp(T value, T min, T max);

/**
 * @brief Requires that the provided size value is odd.
 *
 * This helper is intended for APIs that accept a kernel/window size and build
 * the kernel internally.
 *
 * @param size The size value to validate
 * @throws std::invalid_argument If @p size is even
 */
void requireOddSize(size_t size);

/**
 * @brief Requires that the provided dynamic kernel is square and has odd width.
 *
 * The kernel must be non-empty, each row must have the same size as the number
 * of rows, and the width must be odd.
 *
 * @tparam T The data type of the kernel elements
 * @param kernel A 2D vector kernel to validate
 * @throws std::invalid_argument If @p kernel is empty, not square, or has even width
 */
template<typename T>
void requireSquareAndOdd(const std::vector<std::vector<T>>& kernel);

/**
 * @brief Requires that the provided fixed-size kernel has odd width.
 *
 * A `std::array<std::array<T, N>, N>` kernel is always square by construction,
 * so this overload validates only the odd-width requirement.
 *
 * @tparam T The data type of the kernel elements
 * @tparam N The kernel width and height
 * @param kernel A fixed-size square kernel to validate
 * @throws std::invalid_argument If @p N is even
 */
template<typename T, size_t N>
void requireSquareAndOdd(const std::array<std::array<T, N>, N>& kernel);

/**
 * @brief Requires that two images have matching metadata.
 *
 * This checks every image property except the pixel buffer itself, which makes
 * it suitable for validating input/output image compatibility.
 *
 * @tparam T The data type of the image pixels
 * @param first First image to validate
 * @param second Second image to validate
 * @throws std::invalid_argument If the image metadata does not match
 */
template<typename T>
void requireSameTypeImages(const Image<T>& first, const Image<T>& second);
