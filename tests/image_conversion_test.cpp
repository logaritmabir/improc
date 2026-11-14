#include <gtest/gtest.h>
#include "image_conversion.hpp"
#include <cstdint>

TEST(ImageConversionTest, ExtractChannel) {
    Image<uint8_t> rgb(5, 5, 3);
    rgb.fill(100);
    rgb(0, 0, 0) = 255;
    rgb(0, 0, 1) = 128;
    rgb(0, 0, 2) = 64;

    Image<uint8_t> redChannel(5, 5, 1);
    extractChannel(rgb, redChannel, 0);

    EXPECT_EQ(redChannel(0, 0, 0), 255);
    EXPECT_EQ(redChannel(1, 1, 0), 100);
}

TEST(ImageConversionTest, ExtractChannelRejectsNonRgbImage) {
    Image<uint8_t> grayscale(5, 5, 1);
    Image<uint8_t> output(5, 5, 1);
    EXPECT_THROW(extractChannel(grayscale, output, 0), std::invalid_argument);
}

TEST(ImageConversionTest, ExtractChannelRejectsInvalidChannel) {
    Image<uint8_t> rgb(5, 5, 3);
    Image<uint8_t> output(5, 5, 1);
    EXPECT_THROW(extractChannel(rgb, output, 3), std::invalid_argument);
}

TEST(ImageConversionTest, RgbToGrayscale) {
    Image<uint8_t> rgb(2, 2, 3);
    rgb(0, 0, 0) = 255;
    rgb(0, 0, 1) = 0;
    rgb(0, 0, 2) = 0;

    Image<uint8_t> gray(2, 2, 1);
    rgbToGrayscale(rgb, gray);

    EXPECT_EQ(gray(0, 0, 0), 76);
}

TEST(ImageConversionTest, RgbToGrayscaleRejectsNonRgbImage) {
    Image<uint8_t> grayscale(5, 5, 1);
    Image<uint8_t> output(5, 5, 1);
    EXPECT_THROW(rgbToGrayscale(grayscale, output), std::invalid_argument);
}
