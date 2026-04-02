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

class ImagesEqualTest : public ::testing::Test {
protected:
    void SetUp() override {
        img1 = Image<uint8_t>(3, 3, 1);
        img1.fill(100);
        
        img2 = Image<uint8_t>(3, 3, 1);
        img2.fill(100);
    }
    Image<uint8_t> img1;
    Image<uint8_t> img2;
};

TEST_F(ImagesEqualTest, IdenticalImages) {
    EXPECT_TRUE(imagesEqual(img1, img2));
}

TEST_F(ImagesEqualTest, SameImageWithItself) {
    EXPECT_TRUE(imagesEqual(img1, img1));
}

TEST_F(ImagesEqualTest, DifferentRows) {
    Image<uint8_t> img3(4, 3, 1);
    img3.fill(100);
    EXPECT_FALSE(imagesEqual(img1, img3));
}

TEST_F(ImagesEqualTest, DifferentCols) {
    Image<uint8_t> img3(3, 4, 1);
    img3.fill(100);
    EXPECT_FALSE(imagesEqual(img1, img3));
}

TEST_F(ImagesEqualTest, DifferentChannels) {
    Image<uint8_t> img3(3, 3, 3);
    img3.fill(100);
    EXPECT_FALSE(imagesEqual(img1, img3));
}
