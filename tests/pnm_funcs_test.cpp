#include <gtest/gtest.h>
#include "pnm_funcs.hpp"
#include "image.hpp"
#include <cstdint>
#include <fstream>

class PNMTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::remove("test_output.pgm");
        std::remove("test_output.ppm");
        std::remove("test_specific.pgm");
        std::remove("test_specific.ppm");
        std::remove("test_grayscale.pgm");
        std::remove("test_channel.pgm");
    }
};

TEST_F(PNMTest, SaveAndReadGrayscale) {
    Image<uint8_t> original(10, 10, 1);
    original.fill(128);
    savePNM("test_output.pgm", original, PNMFormat::P5);
    
    Image<uint8_t> loaded;
    readPNM("test_output.pgm", loaded, PNMFormat::P5);
    
    EXPECT_EQ(loaded.rows(), original.rows());
    EXPECT_EQ(loaded.cols(), original.cols());
    EXPECT_EQ(loaded.channels(), original.channels());
    
    for (size_t i = 0; i < original.size(); i++) {
        EXPECT_EQ(loaded.data()[i], original.data()[i]);
    }
}

TEST_F(PNMTest, SaveAndReadMultiChannel) {
    Image<uint8_t> original(5, 5, 3);
    for (size_t r = 0; r < original.rows(); r++) {
        for (size_t c = 0; c < original.cols(); c++) {
            original(r, c, 0) = 100;
            original(r, c, 1) = 150;
            original(r, c, 2) = 200;
        }
    }
    savePNM("test_output.ppm", original, PNMFormat::P6);
    
    Image<uint8_t> loaded;
    readPNM("test_output.ppm", loaded, PNMFormat::P6);
    
    EXPECT_EQ(loaded.channels(), 3u);
    EXPECT_EQ(loaded(0, 0, 0), 100);
    EXPECT_EQ(loaded(0, 0, 1), 150);
    EXPECT_EQ(loaded(0, 0, 2), 200);
}

TEST_F(PNMTest, PgmP5RoundTrip) {
    Image<uint8_t> original(2, 2, 1);
    original.fill(127);
    savePNM("test_specific.pgm", original, PNMFormat::P5);

    Image<uint8_t> loaded;
    readPNM("test_specific.pgm", loaded, PNMFormat::P5);

    EXPECT_EQ(loaded.rows(), 2u);
    EXPECT_EQ(loaded.cols(), 2u);
    EXPECT_EQ(loaded.channels(), 1u);
    EXPECT_EQ(loaded(0, 0, 0), 127);
}

TEST_F(PNMTest, PpmP6RoundTrip) {
    Image<uint8_t> original(2, 2, 3);
    original.fill(127);
    savePNM("test_specific.ppm", original, PNMFormat::P6);

    Image<uint8_t> loaded;
    readPNM("test_specific.ppm", loaded, PNMFormat::P6);

    EXPECT_EQ(loaded.rows(), 2u);
    EXPECT_EQ(loaded.cols(), 2u);
    EXPECT_EQ(loaded.channels(), 3u);
    EXPECT_EQ(loaded(0, 0, 0), 127);
    EXPECT_EQ(loaded(0, 0, 1), 127);
    EXPECT_EQ(loaded(0, 0, 2), 127);
}

TEST_F(PNMTest, SpecificReadersRejectOtherMagicNumber) {
    Image<uint8_t> grayscale(1, 1, 1);
    Image<uint8_t> color(1, 1, 3);
    Image<uint8_t> loaded;
    savePNM("test_specific.pgm", grayscale, PNMFormat::P5);
    savePNM("test_specific.ppm", color, PNMFormat::P6);

    EXPECT_THROW(readPNM("test_specific.ppm", loaded, PNMFormat::P5), std::runtime_error);
    EXPECT_THROW(readPNM("test_specific.pgm", loaded, PNMFormat::P6), std::runtime_error);
}

TEST_F(PNMTest, SavePNMRejectsUnsupportedChannelCount) {
    Image<uint8_t> twoChannel(2, 2, 2);
    EXPECT_THROW(savePNM("test_specific.pgm", twoChannel, PNMFormat::P5), std::invalid_argument);
}

TEST_F(PNMTest, ExtractChannel) {
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

TEST_F(PNMTest, ExtractChannelInvalidImage) {
    Image<uint8_t> grayscale(5, 5, 1);
    Image<uint8_t> output(5, 5, 1);
    EXPECT_THROW(extractChannel(grayscale, output, 0), std::runtime_error);
}

TEST_F(PNMTest, ExtractChannelInvalidChannel) {
    Image<uint8_t> rgb(5, 5, 3);
    Image<uint8_t> output(5, 5, 1);
    EXPECT_THROW(extractChannel(rgb, output, 3), std::runtime_error);
}

TEST_F(PNMTest, RgbToGrayscale) {
    Image<uint8_t> rgb(2, 2, 3);
    rgb(0, 0, 0) = 255; // R
    rgb(0, 0, 1) = 0;   // G
    rgb(0, 0, 2) = 0;   // B
    
    Image<uint8_t> gray(2, 2, 1);
    rgbToGrayscale(rgb, gray);
    
    EXPECT_EQ(gray(0, 0, 0), 76); // 0.299 * 255 ≈ 76
}

TEST_F(PNMTest, RgbToGrayscaleInvalidImage) {
    Image<uint8_t> grayscale(5, 5, 1);
    Image<uint8_t> output(5, 5, 1);
    EXPECT_THROW(rgbToGrayscale(grayscale, output), std::runtime_error);
}

TEST_F(PNMTest, ReadNonexistentFile) {
    Image<uint8_t> img;
    EXPECT_THROW(readPNM("nonexistent.pgm", img, PNMFormat::P5), std::runtime_error);
}

TEST_F(PNMTest, SaveToNonexistentDirectory) {
    Image<uint8_t> img(5, 5, 1);
    img.fill(50);
    EXPECT_THROW(savePNM("/nonexistent/dir/test.pgm", img, PNMFormat::P5), std::runtime_error);
}
