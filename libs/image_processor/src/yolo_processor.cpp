/**
 * @file yolo_processor.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Implementation of the YoloProcessor class for object detection using
 * YOLO models.
 * @version 0.1
 * @date 2025-10-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "yolo_processor.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

bool YoloProcessor::load_model(const YoloConfig& config) {
  // Load class names file and ensure it's not empty
  std::ifstream ifs(config.classes_path);
  if (!ifs) {
    std::cerr << "Error opening classes file: " << config.classes_path
              << std::endl;
    return false;
  }

  std::vector<std::string> classes;
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty()) classes.push_back(line);
  }
  if (classes.empty()) {
    std::cerr << "Classes file is empty: " << config.classes_path << std::endl;
    return false;
  }

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
      std::cerr << "Failed to load model from: " << config.model_path
                << std::endl;
      return false;
    }

    return true;
  } catch (const cv::Exception& e) {
    // model load failed
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return false;
  }
}

YoloResult YoloProcessor::process(const cv::Mat& bgr) {
  YoloResult result;

  if (bgr.empty()) return result;
  if (net_.empty()) return result;

  int w = bgr.cols;
  int h = bgr.rows;
  int maxwh = std::max(w, h);
  cv::Mat input_image = cv::Mat::zeros(maxwh, maxwh, bgr.type());
  bgr.copyTo(input_image(cv::Rect(0, 0, w, h)));

  // Network input size: use 640x640
  const float INPUT_W = config_.net_input.width;
  const float INPUT_H = config_.net_input.height;

  cv::Mat blob;
  cv::dnn::blobFromImage(
      input_image, blob, 1.0 / 255.0,
      cv::Size(static_cast<int>(INPUT_W), static_cast<int>(INPUT_H)),
      cv::Scalar(), true, false);

  net_.setInput(blob);
  std::vector<cv::Mat> outputs;
  try {
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
  } catch (const cv::Exception& e) {
    std::cerr << "Error during YOLO forward pass: " << e.what() << std::endl;
    return result;
  }

  if (outputs.empty()) return result;

  // Use first output blob
  cv::Mat out = outputs[0];

  std::vector<int> class_ids;
  std::vector<float> confidences;
  auto boxes =
      decode(out, config_.confidence_threshold, class_ids, confidences);

  // Scale boxes from network coordinates to original image coordinates.
  float x_factor = static_cast<float>(input_image.cols) / INPUT_W;
  float y_factor = static_cast<float>(input_image.rows) / INPUT_H;

  for (size_t i = 0; i < boxes.size(); ++i) {
    const cv::Rect2f& b = boxes[i];
    float left = b.x * x_factor;
    float top = b.y * y_factor;
    float width = b.width * x_factor;
    float height = b.height * y_factor;

    // Clip to original image size
    left = std::max(0.0f, std::min(left, static_cast<float>(w)));
    top = std::max(0.0f, std::min(top, static_cast<float>(h)));
    if (left + width > w) width = static_cast<float>(w) - left;
    if (top + height > h) height = static_cast<float>(h) - top;

    result.boxes.emplace_back(left, top, width, height);
    result.class_ids.push_back(class_ids[i]);
    result.confidences.push_back(confidences[i]);
  }

  return result;
}

std::vector<cv::Rect2f> YoloProcessor::decode(const cv::Mat& out,
                                              float confidence_threshold,
                                              std::vector<int>& class_ids,
                                              std::vector<float>& confidences) {
  std::vector<cv::Rect2f> boxes;
  class_ids.clear();
  confidences.clear();

  if (out.empty()) return boxes;

  // Expect out to be NxD where D >= 5 (cx,cy,w,h,obj, [class scores...])
  cv::Mat mat = out;
  if (mat.dims == 3) {
    // flatten last two dims if necessary
    mat = mat.reshape(1, mat.size[1]);
  }

  int rows = mat.rows;
  int cols = mat.cols;
  if (rows <= 0 || cols < 5) return boxes;

  std::vector<cv::Rect2f> raw_boxes;
  std::vector<int> raw_class_ids;
  std::vector<float> raw_conf;

  for (int r = 0; r < rows; ++r) {
    const float* data = mat.ptr<float>(r);
    float cx = data[0];
    float cy = data[1];
    float w = data[2];
    float h = data[3];
    float obj = data[4];

    // find best class score if present
    float best_score = 1.0f;
    int best_class = 0;
    if (cols > 5) {
      best_score = -1.0f;
      for (int c = 5; c < cols; ++c) {
        float s = data[c];
        if (s > best_score) {
          best_score = s;
          best_class = c - 5;
        }
      }
    }

    float conf = obj;
    if (cols > 5) conf = obj * best_score;

    if (conf < confidence_threshold) continue;

    float left = cx - w / 2.0f;
    float top = cy - h / 2.0f;

    raw_boxes.emplace_back(left, top, w, h);
    raw_class_ids.push_back(best_class);
    raw_conf.push_back(conf);
  }

  // Apply NMS
  std::vector<int> indices;
  // convert to Rect for NMS (integer rounding)
  std::vector<cv::Rect> rects;
  rects.reserve(raw_boxes.size());
  // Use std::transform to convert floating-point boxes to integer rects
  std::transform(raw_boxes.begin(), raw_boxes.end(), std::back_inserter(rects),
                 [](const cv::Rect2f& rb) {
                   return cv::Rect(static_cast<int>(std::round(rb.x)),
                                   static_cast<int>(std::round(rb.y)),
                                   static_cast<int>(std::round(rb.width)),
                                   static_cast<int>(std::round(rb.height)));
                 });

  if (!rects.empty()) {
    // Use configured score and NMS thresholds
    float score_thr = config_.score_threshold;
    float nms_thr = config_.nms_threshold;
    cv::dnn::NMSBoxes(rects, raw_conf, score_thr, nms_thr, indices);
  }

  if (indices.empty()) {
    // no NMS selected indices but maybe no boxes matched threshold; return raw
    // lists
    for (size_t i = 0; i < raw_boxes.size(); ++i) {
      boxes.push_back(raw_boxes[i]);
      class_ids.push_back(raw_class_ids[i]);
      confidences.push_back(raw_conf[i]);
    }
  } else {
    for (int idx : indices) {
      if (idx >= 0 && static_cast<size_t>(idx) < raw_boxes.size()) {
        boxes.push_back(raw_boxes[idx]);
        class_ids.push_back(raw_class_ids[idx]);
        confidences.push_back(raw_conf[idx]);
      }
    }
  }

  return boxes;
}
