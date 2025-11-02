/**
 * @file depth_processor.hpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief DepthProcessor class for depth estimation from images
 * @version 0.1
 * @date 2025-10-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "image_processor.hpp"

#include <string>


struct DepthConfig {
  std::string model_path;
  cv::Scalar mean{0.485, 0.456, 0.406};
  cv::Scalar std{0.229, 0.224, 0.225};
  double scale{-0.0858};
  double offset{5.808};
  cv::Size net_input{518, 518};
  bool swap_RB{true};
};

struct DepthResult {
  // Colorized depth map for visualization
  cv::Mat visualization;
  // Raw depth map in meters (floating point values)
  cv::Mat depth_map;
};

class DepthProcessor : public ImageProcessor {
 public:
  /**
   * @brief loads the depth estimation model based on the provided
   * configuration.
   *
   * @param config - DepthConfig object containing model parameters.
   * @return true - model loaded successfully
   * @return false - model failed to load
   */
  bool load_model(const DepthConfig& config);

  /**
   * @brief processes a BGR image and returns the depth estimation result.
   *
   * @param bgr - input BGR image as cv::Mat
   * @return DepthResult - result of the depth estimation
   */
  virtual DepthResult process(const cv::Mat& bgr);

 protected:
  /**
   * @brief normalizes the depth map to a 0-1 range.
   *
   * @param d - input depth map as cv::Mat
   * @return cv::Mat - normalized depth map
   */
  static cv::Mat normalize(const cv::Mat& d);

  /**
   * @brief colors the normalized depth map for visualization.
   *
   * @param d01 - input 0-1 normalized depth map as cv::Mat
   * @return cv::Mat - colorized depth map
   */
  static cv::Mat colorize(const cv::Mat& d01);

 private:
  cv::dnn::Net net_;
  DepthConfig config_;
};
