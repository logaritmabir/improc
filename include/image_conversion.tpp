#pragma once

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
