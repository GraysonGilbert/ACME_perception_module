/**
 * @file yolo_processor.hpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief YoloProcessor class for object detection using YOLO models.
 * @version 0.1
 * @date 2025-10-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "image_processor.hpp"

struct YoloConfig {
  std::string model_path;
  std::string classes_path;
  cv::Size net_input{640, 640};
};

struct YoloResult {
  // detected bounding boxes
  std::vector<cv::Rect2f> boxes;
  // corresponding class IDs
  std::vector<int> class_ids;
  // confidence scores for each detection
  std::vector<float> confidences;
};

class YoloProcessor : public ImageProcessor {
 public:
  /**
   * @brief loads the YOLO model based on the provided configuration.
   *
   * @param config - YoloConfig object containing model parameters.
   * @return true - model loaded successfully
   * @return false - model failed to load
   */
  bool load_model(const YoloConfig& config);

  /**
   * @brief processes a BGR image and returns the YOLO detection result.
   *
   * @param bgr - input BGR image as cv::Mat
   * @return ProcessResult - result of the YOLO detection
   */
  virtual YoloResult process(const cv::Mat& bgr);

 protected:
  /**
   * @brief decodes the output of the YOLO model into bounding boxes, class IDs,
   * and confidences.
   *
   * @param out - mat containing the raw output from the YOLO model to decode
   * @param confidence_threshold - threshold to filter weak detections
   * @param class_ids - output vector of class IDs
   * @param confidences - output vector of confidences
   * @return std::vector<cv::Rect2f> - vector of decoded bounding boxes
   */
  std::vector<cv::Rect2f> decode(const cv::Mat& out, float confidence_threshold,
                                 std::vector<int>& class_ids,
                                 std::vector<float>& confidences);

 private:
  cv::dnn::Net net_;
};
