#pragma once

#include <cmath>
#include <type_traits>

template<typename PixelType>
void extractChannel(const Image<PixelType>& input, Image<PixelType>& output, size_t channel) {
    if (input.channels() != 3) {
        throw std::invalid_argument("Image must be RGB (3 channels) to extract a channel");
    }
    if (channel > 2) {
        throw std::invalid_argument("Channel must be 0 (R), 1 (G), or 2 (B)");
    }

    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t col = 0; col < input.cols(); col++) {
            output.at(row, col, 0) = input.at(row, col, channel);
        }
    }
}

template<typename PixelType>
void rgbToGrayscale(const Image<PixelType>& input, Image<PixelType>& output) {
    if (input.channels() != 3) {
        throw std::invalid_argument("Image must be RGB (3 channels)");
    }

    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t col = 0; col < input.cols(); col++) {
            const PixelType red = input.at(row, col, 0);
            const PixelType green = input.at(row, col, 1);
            const PixelType blue = input.at(row, col, 2);
            const PixelType gray = static_cast<PixelType>(
                0.299 * red + 0.587 * green + 0.114 * blue);
            output.at(row, col, 0) = gray;
        }
    }
}

template<typename SourcePixelType, typename DestinationPixelType>
void convertImageType(const Image<SourcePixelType>& input, Image<DestinationPixelType>& output) {
    if (input.rows() != output.rows() ||
        input.cols() != output.cols() ||
        input.channels() != output.channels()) {
        throw std::invalid_argument("Input and output images must have matching dimensions and channels");
    }

    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t col = 0; col < input.cols(); col++) {
            for (size_t ch = 0; ch < input.channels(); ch++) {
                output.at(row, col, ch) = static_cast<DestinationPixelType>(input.at(row, col, ch));
            }
        }
    }
}

template<typename SourcePixelType, typename DestinationPixelType>
void convertImageRange(
    const Image<SourcePixelType>& input,
    Image<DestinationPixelType>& output,
    double sourceMin,
    double sourceMax,
    double destinationMin,
    double destinationMax) {
    static_assert(std::is_arithmetic<SourcePixelType>::value,
        "Source pixel type must be arithmetic");
    static_assert(std::is_arithmetic<DestinationPixelType>::value,
        "Destination pixel type must be arithmetic");

    if (!std::isfinite(sourceMin) || !std::isfinite(sourceMax) ||
        !std::isfinite(destinationMin) || !std::isfinite(destinationMax) ||
        sourceMin >= sourceMax || destinationMin >= destinationMax) {
        throw std::invalid_argument("Image conversion ranges must be finite and ascending");
    }

    const double sourceRange = sourceMax - sourceMin;
    const double destinationRange = destinationMax - destinationMin;
    if (!std::isfinite(sourceRange) || !std::isfinite(destinationRange)) {
        throw std::invalid_argument("Image conversion range widths must be finite");
    }

    if (input.rows() != output.rows() ||
        input.cols() != output.cols() ||
        input.channels() != output.channels()) {
        throw std::invalid_argument("Input and output images must have matching dimensions and channels");
    }

    // TODO: Reconsider whether integral-specific NaN validation and rounding
    // are needed; this check may be removable if the conversion policy changes.
    if (std::is_integral<DestinationPixelType>::value) {
        for (size_t row = 0; row < input.rows(); row++) {
            for (size_t col = 0; col < input.cols(); col++) {
                for (size_t ch = 0; ch < input.channels(); ch++) {
                    if (std::isnan(static_cast<double>(input.at(row, col, ch)))) {
                        throw std::invalid_argument("Cannot convert NaN pixels to an integer type");
                    }
                }
            }
        }
    }

    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t col = 0; col < input.cols(); col++) {
            for (size_t ch = 0; ch < input.channels(); ch++) {
                double value = static_cast<double>(input.at(row, col, ch));
                if (value < sourceMin) {
                    value = sourceMin;
                } else if (value > sourceMax) {
                    value = sourceMax;
                }

                double convertedValue =
                    destinationMin + ((value - sourceMin) / sourceRange) * destinationRange;
                if (convertedValue < destinationMin) {
                    convertedValue = destinationMin;
                } else if (convertedValue > destinationMax) {
                    convertedValue = destinationMax;
                }

                if (std::is_integral<DestinationPixelType>::value) {
                    convertedValue = std::round(convertedValue);
                }
                output.at(row, col, ch) = static_cast<DestinationPixelType>(convertedValue);
            }
        }
    }
}
