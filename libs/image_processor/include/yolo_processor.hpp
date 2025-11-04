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

#pragma once

#include <opencv2/opencv.hpp>

struct YoloConfig {
  std::string model_path;
  std::string classes_path;
  cv::Size net_input{640, 640};
  float score_threshold = 0.2;
  float nms_threshold = 0.4;
  float confidence_threshold = 0.4;
};

struct YoloResult {
  // detected bounding boxes
  std::vector<cv::Rect2f> boxes;
  // corresponding class IDs
  std::vector<int> class_ids;
  // confidence scores for each detection
  std::vector<float> confidences;
};

class YoloProcessor {
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
  YoloConfig config_;
};
