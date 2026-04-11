#include <gtest/gtest.h>
#include "utils.hpp"
#include "image.hpp"
#include <cstdint>

class ClampTest : public ::testing::Test {};

TEST_F(ClampTest, ClampWithinRange) {
    EXPECT_EQ(clamp(5, 0, 10), 5);
    EXPECT_EQ(clamp(0, 0, 10), 0);
    EXPECT_EQ(clamp(10, 0, 10), 10);
}

TEST_F(ClampTest, ClampBelowMin) {
    EXPECT_EQ(clamp(-5, 0, 10), 0);
}

TEST_F(ClampTest, ClampAboveMax) {
    EXPECT_EQ(clamp(15, 0, 10), 10);
}

TEST_F(ClampTest, ClampFloat) {
    EXPECT_EQ(clamp(5.5f, 0.0f, 10.0f), 5.5f);
    EXPECT_EQ(clamp(-1.0f, 0.0f, 10.0f), 0.0f);
    EXPECT_EQ(clamp(11.0f, 0.0f, 10.0f), 10.0f);
}

TEST_F(ClampTest, ClampUint8) {
    EXPECT_EQ(clamp<uint8_t>(100, 0, 255), 100);
    EXPECT_EQ(clamp<uint8_t>(0, 0, 255), 0);
    EXPECT_EQ(clamp<uint8_t>(255, 0, 255), 255);
}

TEST(UtilsValidationTest, RequireOddSizeAcceptsOddValues) {
    EXPECT_NO_THROW(requireOddSize(3));
    EXPECT_NO_THROW(requireOddSize(7));
}

TEST(UtilsValidationTest, RequireOddSizeRejectsEvenValues) {
    EXPECT_THROW(requireOddSize(2), std::invalid_argument);
    EXPECT_THROW(requireOddSize(0), std::invalid_argument);
}

TEST(UtilsValidationTest, RequireSquareAndOddAcceptsSquareOddVectorKernel) {
    const std::vector<std::vector<float>> kernel = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    };

    EXPECT_NO_THROW(requireSquareAndOdd(kernel));
}

TEST(UtilsValidationTest, RequireSquareAndOddRejectsNonSquareVectorKernel) {
    const std::vector<std::vector<float>> kernel = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f}
    };

    EXPECT_THROW(requireSquareAndOdd(kernel), std::invalid_argument);
}

TEST(UtilsValidationTest, RequireSquareAndOddRejectsEvenVectorKernel) {
    const std::vector<std::vector<float>> kernel = {
        {1.0f, 2.0f},
        {2.0f, 1.0f}
    };

    EXPECT_THROW(requireSquareAndOdd(kernel), std::invalid_argument);
}

TEST(UtilsValidationTest, RequireSameTypeImagesAcceptsMatchingImages) {
    Image<uint8_t> img1(3, 3, 1);
    Image<uint8_t> img2(3, 3, 1);

    EXPECT_NO_THROW(requireSameTypeImages(img1, img2));
}

TEST(UtilsValidationTest, RequireSameTypeImagesRejectsDifferentRows) {
    Image<uint8_t> img1(3, 3, 1);
    Image<uint8_t> img2(4, 3, 1);

    EXPECT_THROW(requireSameTypeImages(img1, img2), std::invalid_argument);
}

TEST(UtilsValidationTest, RequireSameTypeImagesRejectsDifferentChannels) {
    Image<uint8_t> img1(3, 3, 1);
    Image<uint8_t> img2(3, 3, 3);

    EXPECT_THROW(requireSameTypeImages(img1, img2), std::invalid_argument);
}
