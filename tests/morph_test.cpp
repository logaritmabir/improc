#include <gtest/gtest.h>
#include "morph.hpp"
#include "image.hpp"
#include <cstdint>

class MorphTest : public ::testing::Test {
protected:
    void SetUp() override {
        img = Image<uint8_t>(3, 3, 1);
        // Create a test pattern:
        // 0 255 0
        // 255 128 255
        // 0 255 0
        img(0, 0, 0) = 0;   img(0, 1, 0) = 255; img(0, 2, 0) = 0;
        img(1, 0, 0) = 255; img(1, 1, 0) = 128; img(1, 2, 0) = 255;
        img(2, 0, 0) = 0;   img(2, 1, 0) = 255; img(2, 2, 0) = 0;
    }
    Image<uint8_t> img;
};

TEST_F(MorphTest, ErodeBasic) {
    Image<uint8_t> output(3, 3, 1);
    erode(img, output, 3);
    
    EXPECT_EQ(output.rows(), img.rows());
    EXPECT_EQ(output.cols(), img.cols());
    EXPECT_EQ(output.channels(), img.channels());
    
    // For erosion with 3x3 kernel, each pixel becomes the minimum in its 3x3 neighborhood
    // Center pixel (1,1): min of all pixels = 0
    // Corner pixels: min of their neighborhoods
    // Edge pixels: min of their neighborhoods
    EXPECT_EQ(output(0, 0, 0), 0); // Corner: min(0,255,255,0,128,255,0,255,0) = 0
    EXPECT_EQ(output(0, 1, 0), 0); // Edge: min(0,255,0,255,128,255,255,128,255) = 0
    EXPECT_EQ(output(0, 2, 0), 0); // Corner: min(255,0,0,128,255,0,255,0,0) = 0
    EXPECT_EQ(output(1, 0, 0), 0); // Edge: min(0,255,255,0,128,0,0,255,255) = 0
    EXPECT_EQ(output(1, 1, 0), 0); // Center: min of all = 0
    EXPECT_EQ(output(1, 2, 0), 0); // Edge: min(255,128,255,0,255,0,255,0,0) = 0
    EXPECT_EQ(output(2, 0, 0), 0); // Corner: min(255,0,255,0,255,128,0,0,255) = 0
    EXPECT_EQ(output(2, 1, 0), 0); // Edge: min(0,255,128,255,0,255,0,0,0) = 0
    EXPECT_EQ(output(2, 2, 0), 0); // Corner: min(128,255,0,255,0,0,0,255,0) = 0
}

TEST_F(MorphTest, DilateBasic) {
    Image<uint8_t> output(3, 3, 1);
    dilate(img, output, 3);
    
    EXPECT_EQ(output.rows(), img.rows());
    EXPECT_EQ(output.cols(), img.cols());
    EXPECT_EQ(output.channels(), img.channels());
    
    // For dilation with 3x3 kernel, each pixel becomes the maximum in its 3x3 neighborhood
    EXPECT_EQ(output(0, 0, 0), 255); // Corner: max(0,255,255,0,128,255,0,255,0) = 255
    EXPECT_EQ(output(0, 1, 0), 255); // Edge: max(0,255,0,255,128,255,255,128,255) = 255
    EXPECT_EQ(output(0, 2, 0), 255); // Corner: max(255,0,0,128,255,0,255,0,0) = 255
    EXPECT_EQ(output(1, 0, 0), 255); // Edge: max(0,255,255,0,128,0,0,255,255) = 255
    EXPECT_EQ(output(1, 1, 0), 255); // Center: max of all = 255
    EXPECT_EQ(output(1, 2, 0), 255); // Edge: max(255,128,255,0,255,0,255,0,0) = 255
    EXPECT_EQ(output(2, 0, 0), 255); // Corner: max(255,0,255,0,255,128,0,0,255) = 255
    EXPECT_EQ(output(2, 1, 0), 255); // Edge: max(0,255,128,255,0,255,0,0,0) = 255
    EXPECT_EQ(output(2, 2, 0), 255); // Corner: max(128,255,0,255,0,0,0,255,0) = 255
}

TEST_F(MorphTest, ErodeEvenKernelSize) {
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(erode(img, output, 4), std::invalid_argument);
}

TEST_F(MorphTest, DilateEvenKernelSize) {
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(dilate(img, output, 4), std::invalid_argument);
}

TEST_F(MorphTest, ErodeZeroKernelSize) {
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(erode(img, output, 0), std::invalid_argument);
}

TEST_F(MorphTest, DilateZeroKernelSize) {
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(dilate(img, output, 0), std::invalid_argument);
}

TEST_F(MorphTest, ErodeDimensionMismatch) {
    Image<uint8_t> wrongSize(2, 2, 1);
    EXPECT_THROW(erode(img, wrongSize, 3), std::invalid_argument);
}

TEST_F(MorphTest, DilateDimensionMismatch) {
    Image<uint8_t> wrongSize(2, 2, 1);
    EXPECT_THROW(dilate(img, wrongSize, 3), std::invalid_argument);
}

TEST_F(MorphTest, ErodeMultiChannelInput) {
    Image<uint8_t> multiChannelImg(3, 3, 3);
    multiChannelImg.fill(128);
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(erode(multiChannelImg, output, 3), std::invalid_argument);
}

TEST_F(MorphTest, DilateMultiChannelInput) {
    Image<uint8_t> multiChannelImg(3, 3, 3);
    multiChannelImg.fill(128);
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(dilate(multiChannelImg, output, 3), std::invalid_argument);
}

TEST_F(MorphTest, ErodeMultiChannelOutput) {
    Image<uint8_t> output(3, 3, 3); // Multi-channel output
    EXPECT_THROW(erode(img, output, 3), std::invalid_argument);
}

TEST_F(MorphTest, DilateMultiChannelOutput) {
    Image<uint8_t> output(3, 3, 3); // Multi-channel output
    EXPECT_THROW(dilate(img, output, 3), std::invalid_argument);
}

TEST_F(MorphTest, ErodeLargerKernel) {
    Image<uint8_t> largerImg(5, 5, 1);
    largerImg.fill(100);
    Image<uint8_t> output(5, 5, 1);
    
    erode(largerImg, output, 5);
    
    // With 5x5 kernel on 5x5 image, all pixels should be 100
    for (size_t r = 0; r < 5; r++) {
        for (size_t c = 0; c < 5; c++) {
            EXPECT_EQ(output(r, c, 0), 100);
        }
    }
}

TEST_F(MorphTest, DilateLargerKernel) {
    Image<uint8_t> largerImg(5, 5, 1);
    largerImg.fill(100);
    Image<uint8_t> output(5, 5, 1);
    
    dilate(largerImg, output, 5);
    
    // With 5x5 kernel on 5x5 image, all pixels should be 100
    for (size_t r = 0; r < 5; r++) {
        for (size_t c = 0; c < 5; c++) {
            EXPECT_EQ(output(r, c, 0), 100);
        }
    }
}