/**
 * @file depth_processor.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief 
 * @version 0.1
 * @date 2025-10-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "depth_processor.hpp"
#include <iostream>

// Stub implementation of DepthProcessor

bool DepthProcessor::load_model(const CommonConfiguration& configuration) {
    std::cout << "[DepthProcessor] load_model() called (stub)" << std::endl;
    configuration_ = static_cast<const DepthConfig&>(configuration);
    return true;  // pretend model loaded successfully
}

ProcessResult DepthProcessor::process(const cv::Mat& bgr) {
    std::cout << "[DepthProcessor] process() called (stub)" << std::endl;

    ProcessResult result;
    result.visualization = bgr.clone();  // Use existing field
    result.raw_primary = bgr.clone();    // Use existing field
    return result;
}

cv::Mat DepthProcessor::normalize(const cv::Mat& d) {
    std::cout << "[DepthProcessor] normalize() called (stub)" << std::endl;
    cv::Mat out;
    return out;
}

cv::Mat DepthProcessor::colorize(const cv::Mat& d01) {
    std::cout << "[DepthProcessor] colorize() called (stub)" << std::endl;
    cv::Mat colorized;
    return colorized;
}
