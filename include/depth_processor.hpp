// DepthProcessor.hpp
#pragma once
#include "image_processor.hpp"

struct DepthConfig : public CommonConfiguration {
  // depth-anything defaults; override as needed
  cv::Scalar mean{0.485, 0.456, 0.406};
  cv::Scalar std{0.229, 0.224, 0.225};
  cv::Size   net_input{518, 518};
  bool swap_RB{true};
  double scale{1.0};
};

class DepthProcessor : public ImageProcessor {
public:
  /**
   * @brief loads the depth estimation model based on the provided configuration.
   * 
   * @param configuration - DepthConfig object containing model parameters.
   * @return true - model loaded successfully
   * @return false - model failed to load
   */
  bool load_model(const CommonConfiguration& configuration) override;

  /**
   * @brief processes a BGR image and returns the depth estimation result.
   * 
   * @param bgr - input BGR image as cv::Mat
   * @return ProcessResult - result of the depth estimation
   */
  ProcessResult process(const cv::Mat& bgr) override;

private:
  DepthConfig configuration_;
  cv::dnn::Net net_;
  bool with_letterbox_{true};

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
};
