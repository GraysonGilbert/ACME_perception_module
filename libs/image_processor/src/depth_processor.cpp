/**
 * @file depth_processor.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Implementation of DepthProcessor class for depth estimation from images
 * @version 0.1
 * @date 2025-10-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "depth_processor.hpp"
#include <fstream>
#include <string>
#include <iostream>

bool DepthProcessor::load_model(const DepthConfig& config) {
  // Load depth model and ensure it's not empty
  if (config.model_path.empty()){
    std::cerr << "Model path is empty: " << config.model_path << '\n';
    return false;
  }

  try {
    // Read network from file. Use OpenCV DNN default (CPU) backend/target.
    cv::dnn::Net net = cv::dnn::readNet(config.model_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // assign to member
    net_ = net;

    // Ensure net loaded
    if (net_.empty()) {
      std::cerr << "Failed to load model from: " << config.model_path << '\n';
      return false;
    }

    return true;
  } catch (const cv::Exception& e) {
    // model load failed
    std::cerr << "Error loading model: " << e.what() << '\n';
    return false;
  }
  
}

cv::Mat DepthProcessor::normalize(const cv::Mat& d) {
  // TODO: Implement normalization logic
  cv::Mat out;
  return out;
}

cv::Mat DepthProcessor::colorize(const cv::Mat& d01) {
  // TODO: Implement colorization logic
  cv::Mat color;
  return color;
}

DepthResult DepthProcessor::process(const cv::Mat& bgr) {
  // TODO: Implement depth processing logic
  DepthResult r;
  return r;
}
