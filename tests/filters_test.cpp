#include <gtest/gtest.h>
#include "filters.hpp"
#include "image.hpp"
#include <cstdint>
#include <vector>
#include <array>

class FiltersTest : public ::testing::Test {
protected:
    void SetUp() override {
        img = Image<uint8_t>(3, 3, 1);
        img.fill(10);
    }
    Image<uint8_t> img;
};

TEST_F(FiltersTest, ConvolveBoxBlur) {
    std::vector<std::vector<float>> kernel = {
        {1.0f/9, 1.0f/9, 1.0f/9},
        {1.0f/9, 1.0f/9, 1.0f/9},
        {1.0f/9, 1.0f/9, 1.0f/9}
    };
    
    Image<uint8_t> output(3, 3, 1);
    convolve(img, output, kernel);
    
    EXPECT_EQ(output.rows(), img.rows());
    EXPECT_EQ(output.cols(), img.cols());
}

TEST_F(FiltersTest, ConvolveBoxBlurWithArray) {
    constexpr std::array<std::array<float, 3>, 3> kernel = {{
        {{1.0f/9, 1.0f/9, 1.0f/9}},
        {{1.0f/9, 1.0f/9, 1.0f/9}},
        {{1.0f/9, 1.0f/9, 1.0f/9}}
    }};
    
    Image<uint8_t> output(3, 3, 1);
    convolve(img, output, kernel);
    
    EXPECT_EQ(output.rows(), img.rows());
    EXPECT_EQ(output.cols(), img.cols());
}

TEST_F(FiltersTest, ConvolveEmptyKernel) {
    Image<uint8_t> output(3, 3, 1);
    std::vector<std::vector<float>> emptyKernel;
    EXPECT_THROW(convolve(img, output, emptyKernel), std::invalid_argument);
}

TEST_F(FiltersTest, ConvolveNonSquareKernel) {
    Image<uint8_t> output(3, 3, 1);
    std::vector<std::vector<float>> nonSquare = {{1, 2, 3}, {4, 5, 6}};
    EXPECT_THROW(convolve(img, output, nonSquare), std::invalid_argument);
}

TEST_F(FiltersTest, ConvolveEvenKernelSize) {
    Image<uint8_t> output(3, 3, 1);
    std::vector<std::vector<float>> evenKernel = {{1, 2}, {3, 4}};
    EXPECT_THROW(convolve(img, output, evenKernel), std::invalid_argument);
}

TEST_F(FiltersTest, ConvolveDimensionMismatch) {
    Image<uint8_t> wrongSize(2, 2, 1);
    std::vector<std::vector<float>> kernel = {{1, 1}, {1, 1}};
    EXPECT_THROW(convolve(img, wrongSize, kernel), std::invalid_argument);
}

TEST_F(FiltersTest, MedianFilter) {
    Image<uint8_t> output(3, 3, 1);
    medianFilter(img, output, 3);
    
    EXPECT_EQ(output.rows(), img.rows());
    EXPECT_EQ(output.cols(), img.cols());
}

TEST_F(FiltersTest, MedianFilterInvalidSize) {
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(medianFilter(img, output, 2), std::invalid_argument);
    EXPECT_THROW(medianFilter(img, output, 1), std::invalid_argument);
}

TEST_F(FiltersTest, GaussianBlur) {
    Image<uint8_t> output(3, 3, 1);
    gaussianBlur(img, output, 1.0f);
    
    EXPECT_EQ(output.rows(), img.rows());
    EXPECT_EQ(output.cols(), img.cols());
}

TEST_F(FiltersTest, GaussianBlurInvalidSigma) {
    Image<uint8_t> output(3, 3, 1);
    EXPECT_THROW(gaussianBlur(img, output, 0.0f), std::invalid_argument);
    EXPECT_THROW(gaussianBlur(img, output, -1.0f), std::invalid_argument);
}

TEST_F(FiltersTest, SobelFilter) {
    Image<uint8_t> output(3, 3, 1);
    sobelFilter(img, output);
    
    EXPECT_EQ(output.rows(), img.rows());
    EXPECT_EQ(output.cols(), img.cols());
}

TEST_F(FiltersTest, SobelFilterDimensionMismatch) {
    Image<uint8_t> wrongSize(2, 2, 1);
    EXPECT_THROW(sobelFilter(img, wrongSize), std::invalid_argument);
}
