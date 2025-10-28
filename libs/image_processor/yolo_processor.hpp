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
  /**
   * @brief loads the YOLO model based on the provided configuration.
   * 
   * @param configuration - YoloConfig object containing model parameters.
   * @return true - model loaded successfully
   * @return false - model failed to load
   */
  bool load_model(const CommonConfiguration& configuration) override;

  /**
   * @brief processes a BGR image and returns the YOLO detection result.
   * 
   * @param bgr - input BGR image as cv::Mat
   * @return ProcessResult - result of the YOLO detection
   */
  ProcessResult process(const cv::Mat& bgr) override;

private:
  YoloConfig configuration_;
  cv::dnn::Net net_;
  bool with_letterbox_{true};
  cv::Size input_size_;
  std::vector<std::string> output_names_;

  // helpers
  /**
   * @brief decodes the output of the YOLO model into bounding boxes, class IDs, and confidences.
   * 
   * @param out - output cv::Mat from the YOLO model
   * @param confidence_threshold - threshold to filter weak detections
   * @param class_ids - output vector of class IDs
   * @param confidences - output vector of confidences
   * @param letterbox_metadata - optional LetterboxMeta for unletterboxing boxes
   * @return std::vector<cv::Rect2f> - vector of decoded bounding boxes
   */
  std::vector<cv::Rect2f> decode(const cv::Mat& out, float confidence_threshold,
                                 std::vector<int>& class_ids,
                                 std::vector<float>& confidences,
                                 const LetterboxMeta* letterbox_metadata);

  /**
   * @brief draws the detection results on the input frame.
   * 
   * @param frame - input frame as cv::Mat
   * @param boxes - vector of bounding boxes
   * @param class_ids - vector of class IDs
   * @param confidences - vector of confidences
   * @param names - vector of class names
   * @return cv::Mat - frame with drawn detections
   */
  cv::Mat draw(const cv::Mat& frame, const std::vector<cv::Rect2f>& boxes,
               const std::vector<int>& class_ids, const std::vector<float>& confidences,
               const std::vector<std::string>& names);
};
