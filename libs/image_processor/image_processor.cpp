/**
 * @file image_processor.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief 
 * @version 0.1
 * @date 2025-10-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */


#include "image_processor.hpp"
#include <iostream>

// Stub derived class just for linking/testing purposes
class StubImageProcessor : public ImageProcessor {
public:
    bool load_model(const CommonConfiguration& configuration) override {
        std::cout << "[StubImageProcessor] load_model() called (stub)" << std::endl;
        (void)configuration;  // silence unused warning
        return true;  // pretend model loaded successfully
    }

    ProcessResult process(const cv::Mat& bgr) override {
        std::cout << "[StubImageProcessor] process() called (stub)" << std::endl;
        ProcessResult result;
        result.visualization = bgr.clone(); // dummy copy
        result.raw_primary = bgr.clone();   // dummy copy
        return result;
    }
};

