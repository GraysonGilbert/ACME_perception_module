// YoloProcessor.hpp
#pragma once
#include "image_processor.hpp"
#include "letterbox.hpp"
#include <opencv2/opencv.hpp>

struct YoloConfig : public CommonConfiguration {
  float confidence_threshold{0.4f};
  float nms_threshold{0.4f};
  std::vector<std::string> class_names;
  std::string classes_path;
};

struct YoloResult : public ProcessResult {
  std::vector<cv::Rect2f> boxes;
  std::vector<int> class_ids;
  std::vector<float> confidences;
};

class YoloProcessor : public ImageProcessor {
public:
  bool load_model(const CommonConfiguration& configuration) override;
  ProcessResult process(const cv::Mat& bgr) override;

private:
  YoloConfig configuration_;
  cv::dnn::Net net_;
  bool with_letterbox_{true};
  cv::Size input_size_;
  std::vector<std::string> output_names_;

  // helpers
  std::vector<cv::Rect2f> decode(const cv::Mat& out, float confidence_threshold,
                                 std::vector<int>& class_ids,
                                 std::vector<float>& confidences,
                                 const LetterboxMeta* letterbox_metadata);
  cv::Mat draw(const cv::Mat& frame, const std::vector<cv::Rect2f>& boxes,
               const std::vector<int>& class_ids, const std::vector<float>& confidences,
               const std::vector<std::string>& names);
};
