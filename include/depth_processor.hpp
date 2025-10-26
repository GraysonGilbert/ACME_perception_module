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
  bool load_model(const CommonConfiguration& configuration) override;
  ProcessResult process(const cv::Mat& bgr) override;

private:
  DepthConfig configuration_;
  cv::dnn::Net net_;
  bool with_letterbox_{true};

  static cv::Mat normalize(const cv::Mat& d);
  static cv::Mat colorize(const cv::Mat& d01);
};
