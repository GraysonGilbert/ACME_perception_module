/**
 * @file model_test.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief 
 * @version 0.1
 * @date 2025-10-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */

 #include <gtest/gtest.h>
 #include "image_processor.hpp"



 TEST(ImageProcessorTest, InterfaceImplementation) {
    class DummyProcessor : public ImageProcessor {
    public:
        bool load_model(const CommonConfiguration&) override { return true; }
        ProcessResult process(const cv::Mat&) override { return {}; }
    };
    DummyProcessor proc;
    EXPECT_TRUE(proc.load_model(CommonConfiguration{}));
}