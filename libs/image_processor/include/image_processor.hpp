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
