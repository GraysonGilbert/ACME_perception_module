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

/*
 * MIT License
 * 
 * Copyright (c) 2025 Grayson G. & Marcus Hurt
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "depth_processor.hpp"

#include <iostream>
#include <string>

#include "letterbox.hpp"

bool DepthProcessor::load_model(const DepthConfig& config) {
  // Load depth model and ensure it's not empty
  if (config.model_path.empty()) {
    std::cerr << "Model path is empty: " << config.model_path << '\n';
    return false;
  }

  // Assign config to member
  config_ = config;

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
  // Return empty matrix if input is empty
  if (d.empty()) {
    return cv::Mat();
  }

  cv::Mat out;
  double minVal, maxVal;

  cv::minMaxLoc(d, &minVal, &maxVal);  // Find min and max values in input

  if (minVal == maxVal) {
    // If all values are the same, return a zero matrix
    out = cv::Mat::zeros(d.size(), CV_32F);
  } else {
    // Normalize values to the range [0, 1]
    out = (d - minVal) / (maxVal - minVal);
  }
  return out;
}

cv::Mat DepthProcessor::colorize(const cv::Mat& d01) {
  // Return empty matrix if input is empty
  if (d01.empty()) {
    return cv::Mat();
  }

  cv::Mat d01_inv = 1.0 - d01;

  // Convert to input to 8-bit (0-255)
  cv::Mat d8U;
  d01_inv.convertTo(d8U, CV_8U, 255.0);

  // Apply a colormap to generate a color image
  cv::Mat color;
  cv::applyColorMap(d8U, color, cv::COLORMAP_JET);

  return color;
}

DepthResult DepthProcessor::process(const cv::Mat& bgr) {
  // Check for empty frame or model
  DepthResult result;
  if (bgr.empty() || net_.empty()) {
    return result;
  }

  // Letterbox frame
  cv::Mat input;
  LetterboxMeta meta = letterbox(bgr, input, config_.net_input);

  // Create Blob
  cv::Mat blob;
  cv::dnn::blobFromImage(input, blob, (1.0 / 255.0), config_.net_input,
                         config_.mean, config_.swap_RB, false);
  net_.setInput(blob);

  // Inference frame using model
  cv::Mat depth_prediction = net_.forward();
  cv::Mat depth_2d = depth_prediction.reshape(1, depth_prediction.size[2]);

  // Post-process blob to unletterbox
  result.depth_map = unletterboxDepth(depth_2d, meta);

  // Calibrate entire depth map
  result.depth_map = (result.depth_map * config_.scale) + config_.offset;
  cv::threshold(result.depth_map, result.depth_map, 0.0, 0.0,
                cv::THRESH_TOZERO);  // remove any negative outliers

  // Normalize and colorize
  cv::Mat depth_norm = normalize(result.depth_map);
  cv::Mat depth_color = colorize(depth_norm);
  result.visualization = depth_color;

  return result;
}
