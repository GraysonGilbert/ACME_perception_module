#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

struct CommonConfiguration {
  std::string model_path;
  bool prefer_cuda{true};
  bool use_letterbox{true};
  cv::Size net_input{640, 640};
  cv::Scalar mean{0,0,0};
  cv::Scalar std{1,1,1};
  double scale{1.0};
  bool swap_RB{true};
};

struct ProcessResult {
  cv::Mat visualization;
  cv::Mat raw_primary;
};

class ImageProcessor {
public:
  virtual ~ImageProcessor() = default;

  virtual bool load_model(const CommonConfiguration& configuration) = 0;

  virtual ProcessResult process(const cv::Mat& bgr) = 0;

  virtual cv::Mat render(const ProcessResult& result) { return result.visualization; }

protected:
  static void setBackend(cv::dnn::Net& net, bool prefer_cuda) {
#ifdef HAVE_OPENCV_DNN
    if (prefer_cuda) {
#if CV_VERSION_MAJOR >= 4
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
    } else {
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
#endif
  }
};
