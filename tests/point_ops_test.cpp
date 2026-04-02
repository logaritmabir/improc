#include <gtest/gtest.h>
#include "point_ops.hpp"
#include "image.hpp"
#include <cstdint>

class PointOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        img1 = Image<uint8_t>(3, 3, 1);
        img1.fill(100);
        
        img2 = Image<uint8_t>(3, 3, 3);
        img2.fill(50);
    }
    Image<uint8_t> img1;
    Image<uint8_t> img2;
};

TEST_F(PointOpsTest, InvertGrayscale) {
    Image<uint8_t> output(3, 3, 1);
    invert(img1, output);
    
    for (size_t r = 0; r < img1.rows(); r++) {
        for (size_t c = 0; c < img1.cols(); c++) {
            EXPECT_EQ(output(r, c, 0), 155); // 255 - 100 = 155
        }
    }
}

TEST_F(PointOpsTest, InvertMultiChannel) {
    Image<uint8_t> output(3, 3, 3);
    invert(img2, output);
    
    for (size_t i = 0; i < img2.size(); i++) {
        EXPECT_EQ(output.data()[i], 205); // 255 - 50 = 205
    }
}

TEST_F(PointOpsTest, InvertDimensionMismatch) {
    Image<uint8_t> wrongSize(2, 2, 1);
    EXPECT_THROW(invert(img1, wrongSize), std::invalid_argument);
}

TEST_F(PointOpsTest, ThresholdAllAbove) {
    Image<uint8_t> output(3, 3, 1);
    threshold(img1, output, (uint8_t)50);
    
    for (size_t r = 0; r < img1.rows(); r++) {
        for (size_t c = 0; c < img1.cols(); c++) {
            EXPECT_EQ(output(r, c, 0), 255); // 100 >= 50
        }
    }
}

TEST_F(PointOpsTest, ThresholdAllBelow) {
    Image<uint8_t> output(3, 3, 1);
    threshold(img1, output, (uint8_t)200);
    
    for (size_t r = 0; r < img1.rows(); r++) {
        for (size_t c = 0; c < img1.cols(); c++) {
            EXPECT_EQ(output(r, c, 0), 0); // 100 < 200
        }
    }
}

TEST_F(PointOpsTest, ThresholdPartial) {
    Image<uint8_t> img(3, 3, 1);
    img.fill(100);
    img(1, 1, 0) = 200;
    
    Image<uint8_t> output(3, 3, 1);
    threshold(img, output, (uint8_t)150);
    
    EXPECT_EQ(output(0, 0, 0), 0);
    EXPECT_EQ(output(1, 1, 0), 255);
}

TEST_F(PointOpsTest, ThresholdDimensionMismatch) {
    Image<uint8_t> wrongSize(2, 2, 1);
    EXPECT_THROW(threshold(img1, wrongSize, (uint8_t)128), std::invalid_argument);
}

TEST_F(PointOpsTest, ThresholdMultiChannelInput) {
    Image<uint8_t> output(3, 3, 3);
    EXPECT_THROW(threshold(img2, output, (uint8_t)128), std::invalid_argument);
}

TEST_F(PointOpsTest, ThresholdMultiChannelOutput) {
    Image<uint8_t> output(3, 3, 3);
    EXPECT_THROW(threshold(img1, output, (uint8_t)128), std::invalid_argument);
}
