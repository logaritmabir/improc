#pragma once

/**
 * @file utils.hpp
 * @brief Utility functions for image processing operations.
 */

#include <cstdint>

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
