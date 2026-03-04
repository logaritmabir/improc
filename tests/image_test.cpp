#include <gtest/gtest.h>
#include "image.hpp"
#include <cstdint>

class ImageTest : public ::testing::Test {
protected:
    Image<uint8_t> img;
};

TEST_F(ImageTest, DefaultConstructor) {
    EXPECT_EQ(img.rows(), 0);
    EXPECT_EQ(img.cols(), 0);
    EXPECT_EQ(img.channels(), 0);
    EXPECT_EQ(img.stride(), 0);
    EXPECT_TRUE(img.empty());
}

TEST_F(ImageTest, ParameterizedConstructor) {
    Image<uint8_t> img2(10, 20, 3);
    EXPECT_EQ(img2.rows(), 10);
    EXPECT_EQ(img2.cols(), 20);
    EXPECT_EQ(img2.channels(), 3);
    EXPECT_EQ(img2.stride(), 60);
    EXPECT_EQ(img2.size(), 600);
    EXPECT_FALSE(img2.empty());
}

TEST_F(ImageTest, ParameterizedConstructorWithSingleChannel) {
    Image<uint8_t> gray(5, 5, 1);
    EXPECT_EQ(gray.rows(), 5);
    EXPECT_EQ(gray.cols(), 5);
    EXPECT_EQ(gray.channels(), 1);
    EXPECT_EQ(gray.stride(), 5);
}

TEST_F(ImageTest, ParameterizedConstructorInvalidRows) {
    EXPECT_THROW(Image<uint8_t>(0, 10, 3), std::invalid_argument);
}

TEST_F(ImageTest, ParameterizedConstructorInvalidCols) {
    EXPECT_THROW(Image<uint8_t>(10, 0, 3), std::invalid_argument);
}

TEST_F(ImageTest, ParameterizedConstructorInvalidChannels) {
    EXPECT_THROW(Image<uint8_t>(10, 10, 0), std::invalid_argument);
}

TEST_F(ImageTest, CopyConstructor) {
    Image<uint8_t> img1(5, 5, 3);
    img1.fill(42);
    
    Image<uint8_t> img2(img1);
    EXPECT_EQ(img2.rows(), 5);
    EXPECT_EQ(img2.cols(), 5);
    EXPECT_EQ(img2.channels(), 3);
    EXPECT_EQ(img2.size(), img1.size());
    EXPECT_EQ(img2(0, 0, 0), 42);
}

TEST_F(ImageTest, MoveConstructor) {
    Image<uint8_t> img1(5, 5, 3);
    img1.fill(123);
    size_t original_size = img1.size();
    
    Image<uint8_t> img2(std::move(img1));
    EXPECT_EQ(img2.rows(), 5);
    EXPECT_EQ(img2.cols(), 5);
    EXPECT_EQ(img2.size(), original_size);
    EXPECT_EQ(img2(0, 0, 0), 123);
    
    EXPECT_EQ(img1.rows(), 0);
    EXPECT_EQ(img1.cols(), 0);
    EXPECT_TRUE(img1.empty());
}

TEST_F(ImageTest, Accessors) {
    Image<uint8_t> img2(4, 6, 2);
    EXPECT_EQ(img2.rows(), 4);
    EXPECT_EQ(img2.cols(), 6);
    EXPECT_EQ(img2.channels(), 2);
    EXPECT_EQ(img2.stride(), 12);
}

TEST_F(ImageTest, AtMethodRead) {
    Image<uint8_t> img2(3, 3, 1);
    img2.fill(100);
    EXPECT_EQ(img2.at(0, 0, 0), 100);
    EXPECT_EQ(img2.at(2, 2, 0), 100);
}

TEST_F(ImageTest, AtMethodWrite) {
    Image<uint8_t> img2(3, 3, 3);
    img2.at(1, 1, 0) = 50;
    img2.at(1, 1, 1) = 100;
    img2.at(1, 1, 2) = 200;
    
    EXPECT_EQ(img2.at(1, 1, 0), 50);
    EXPECT_EQ(img2.at(1, 1, 1), 100);
    EXPECT_EQ(img2.at(1, 1, 2), 200);
}

TEST_F(ImageTest, AtMethodOutOfRangeRow) {
    Image<uint8_t> img2(5, 5, 1);
    EXPECT_THROW(img2.at(5, 0, 0), std::out_of_range);
}

TEST_F(ImageTest, AtMethodOutOfRangeCol) {
    Image<uint8_t> img2(5, 5, 1);
    EXPECT_THROW(img2.at(0, 5, 0), std::out_of_range);
}

TEST_F(ImageTest, AtMethodOutOfRangeChannel) {
    Image<uint8_t> img2(5, 5, 3);
    EXPECT_THROW(img2.at(0, 0, 3), std::out_of_range);
}

TEST_F(ImageTest, OperatorParentheses) {
    Image<uint8_t> img2(4, 4, 2);
    img2(1, 1, 0) = 75;
    img2(1, 1, 1) = 150;
    
    EXPECT_EQ(img2(1, 1, 0), 75);
    EXPECT_EQ(img2(1, 1, 1), 150);
}

TEST_F(ImageTest, DataPointer) {
    Image<uint8_t> img2(2, 2, 1);
    uint8_t* ptr = img2.data();
    EXPECT_NE(ptr, nullptr);
    
    ptr[0] = 42;
    EXPECT_EQ(img2(0, 0, 0), 42);
}

TEST_F(ImageTest, ConstDataPointer) {
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(88);
    const uint8_t* ptr = img2.data();
    EXPECT_EQ(ptr[0], 88);
}

TEST_F(ImageTest, Fill) {
    Image<uint8_t> img2(3, 3, 3);
    img2.fill(255);
    
    for (size_t i = 0; i < img2.size(); ++i) {
        EXPECT_EQ(img2.data()[i], 255);
    }
}

TEST_F(ImageTest, Clear) {
    Image<uint8_t> img2(5, 5, 3);
    img2.fill(100);
    img2.clear();

    EXPECT_EQ(img2.rows(), 0);
    EXPECT_EQ(img2.cols(), 0);
    EXPECT_EQ(img2.channels(), 0);
    EXPECT_TRUE(img2.empty());
}

TEST_F(ImageTest, Resize) {
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(50);

    img2.resize(4, 4, 3);
    EXPECT_EQ(img2.rows(), 4);
    EXPECT_EQ(img2.cols(), 4);
    EXPECT_EQ(img2.channels(), 3);
    EXPECT_EQ(img2.size(), 48);
}

TEST_F(ImageTest, ResizeInvalidDimensions) {
    Image<uint8_t> img2(5, 5, 1);
    EXPECT_THROW(img2.resize(0, 5, 1), std::invalid_argument);
    EXPECT_THROW(img2.resize(5, 0, 1), std::invalid_argument);
    EXPECT_THROW(img2.resize(5, 5, 0), std::invalid_argument);
}

TEST_F(ImageTest, CopyAssignment) {
    Image<uint8_t> img1(3, 3, 1);
    img1.fill(77);
    
    Image<uint8_t> img2(2, 2, 2);
    img2 = img1;
    
    EXPECT_EQ(img2.rows(), 3);
    EXPECT_EQ(img2.cols(), 3);
    EXPECT_EQ(img2.channels(), 1);
    EXPECT_EQ(img2(0, 0, 0), 77);
}

TEST_F(ImageTest, CopyAssignmentSelf) {
    Image<uint8_t> img1(3, 3, 1);
    img1.fill(99);
    
    img1 = img1;
    EXPECT_EQ(img1.rows(), 3);
    EXPECT_EQ(img1.cols(), 3);
    EXPECT_EQ(img1(0, 0, 0), 99);
}

TEST_F(ImageTest, MoveAssignment) {
    Image<uint8_t> img1(3, 3, 2);
    img1.fill(111);
    
    Image<uint8_t> img2;
    img2 = std::move(img1);
    
    EXPECT_EQ(img2.rows(), 3);
    EXPECT_EQ(img2.cols(), 3);
    EXPECT_EQ(img2.channels(), 2);
    EXPECT_EQ(img2(0, 0, 0), 111);
    
    EXPECT_EQ(img1.rows(), 0);
}

TEST_F(ImageTest, MoveAssignmentSelf) {
    Image<uint8_t> img1(3, 3, 1);
    img1.fill(55);
    
    img1 = std::move(img1);
    EXPECT_EQ(img1.rows(), 3);
    EXPECT_EQ(img1.cols(), 3);
}

TEST_F(ImageTest, EqualityOperator) {
    Image<uint8_t> img1(2, 2, 1);
    img1.fill(50);
    
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(50);
    
    EXPECT_TRUE(img1 == img2);
}

TEST_F(ImageTest, EqualityOperatorDifferentDimensions) {
    Image<uint8_t> img1(2, 2, 1);
    Image<uint8_t> img2(3, 3, 1);
    
    EXPECT_FALSE(img1 == img2);
}

TEST_F(ImageTest, EqualityOperatorDifferentData) {
    Image<uint8_t> img1(2, 2, 1);
    img1.fill(50);
    
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(100);
    
    EXPECT_FALSE(img1 == img2);
}

TEST_F(ImageTest, InequalityOperator) {
    Image<uint8_t> img1(2, 2, 1);
    img1.fill(50);
    
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(100);
    
    EXPECT_TRUE(img1 != img2);
}

TEST_F(ImageTest, InequalityOperatorFalse) {
    Image<uint8_t> img1(2, 2, 1);
    img1.fill(50);
    
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(50);
    
    EXPECT_FALSE(img1 != img2);
}

TEST_F(ImageTest, IteratorBeginEnd) {
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(77);
    
    int count = 0;
    for (auto it = img2.begin(); it != img2.end(); ++it) {
        EXPECT_EQ(*it, 77);
        count++;
    }
    EXPECT_EQ(count, 4);
}

TEST_F(ImageTest, ConstIteratorBeginEnd) {
    Image<uint8_t> img2(2, 2, 1);
    img2.fill(88);
    
    const Image<uint8_t>& img_ref = img2;
    int count = 0;
    for (auto it = img_ref.begin(); it != img_ref.end(); ++it) {
        EXPECT_EQ(*it, 88);
        count++;
    }
    EXPECT_EQ(count, 4);
}

TEST_F(ImageTest, CbeginCend) {
    Image<uint8_t> img2(3, 3, 1);
    img2.fill(99);
    
    int count = 0;
    for (auto it = img2.cbegin(); it != img2.cend(); ++it) {
        EXPECT_EQ(*it, 99);
        count++;
    }
    EXPECT_EQ(count, 9);
}

TEST_F(ImageTest, RangeBasedForLoop) {
    Image<uint8_t> img2(2, 3, 1);
    img2.fill(42);
    
    int count = 0;
    for (auto val : img2) {
        EXPECT_EQ(val, 42);
        count++;
    }
    EXPECT_EQ(count, 6);
}

TEST_F(ImageTest, MultiChannelAccess) {
    Image<uint8_t> img2(2, 2, 3);
    img2(0, 0, 0) = 100;  // R
    img2(0, 0, 1) = 150;  // G
    img2(0, 0, 2) = 200;  // B
    
    EXPECT_EQ(img2.at(0, 0, 0), 100);
    EXPECT_EQ(img2.at(0, 0, 1), 150);
    EXPECT_EQ(img2.at(0, 0, 2), 200);
}

TEST_F(ImageTest, SizeCalculation) {
    Image<uint8_t> img2(10, 20, 3);
    EXPECT_EQ(img2.size(), 600);
    
    Image<uint8_t> img3(5, 5, 1);
    EXPECT_EQ(img3.size(), 25);
}

TEST_F(ImageTest, StrideCalculation) {
    Image<uint8_t> img2(10, 20, 3);
    EXPECT_EQ(img2.stride(), 60);
    
    Image<uint8_t> img3(10, 20, 1);
    EXPECT_EQ(img3.stride(), 20);
}
