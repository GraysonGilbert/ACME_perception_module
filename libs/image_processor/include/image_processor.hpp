/**
 * @file image_processor.hpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Parent class for image processors
 * @version 0.1
 * @date 2025-10-30
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <opencv2/opencv.hpp>

class ImageProcessor {
 public:
  virtual ~ImageProcessor() = default;

 protected:
  static void set_backend(cv::dnn::Net& net) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
};
