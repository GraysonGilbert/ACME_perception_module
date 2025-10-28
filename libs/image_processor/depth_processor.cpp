// DepthProcessor.cpp
#include "depth_processor.hpp"
#include "letterbox.hpp"

bool DepthProcessor::load_model(const CommonConfiguration& baseconfiguration) {
  configuration_ = static_cast<const DepthConfig&>(baseconfiguration);
  with_letterbox_ = configuration_.use_letterbox;

  net_ = cv::dnn::readNetFromONNX(configuration_.model_path);
  set_backend(net_, configuration_.prefer_cuda);
  return true;
}

cv::Mat DepthProcessor::normalize(const cv::Mat& d) {
  CV_Assert(d.type() == CV_32F || d.type() == CV_16U || d.type() == CV_8U);
  cv::Mat f; d.convertTo(f, CV_32F);
  double mn, mx; cv::minMaxLoc(f, &mn, &mx);
  if (mx <= mn) mx = mn + 1.0;
  cv::Mat out; cv::subtract(f, mn, out); out /= float(mx - mn);
  return out;
}

cv::Mat DepthProcessor::colorize(const cv::Mat& d01) {
  cv::Mat u8; d01.convertTo(u8, CV_8U, 255.0);
  cv::Mat color; cv::applyColorMap(u8, color, cv::COLORMAP_TURBO);
  return color;
}

ProcessResult DepthProcessor::process(const cv::Mat& bgr) {
  cv::Mat input;
  LetterboxMeta meta;
  if (with_letterbox_) {
    meta = letterbox(bgr, input, configuration_.net_input, {0,0,0});
  } else {
    cv::resize(bgr, input, configuration_.net_input);
  }

  // Standard ImageNet normalization (as in your depth.cpp)
  cv::Mat f32; input.convertTo(f32, CV_32F, 1.0/255.0);
  std::vector<cv::Mat> ch; cv::split(f32, ch);
  for (int i = 0; i < 3; ++i) ch[i] = (ch[i] - configuration_.mean[i]) / configuration_.std[i];
  cv::merge(ch, f32);

  cv::Mat blob = cv::dnn::blobFromImage(f32, 1.0, configuration_.net_input, {0,0,0}, configuration_.swap_RB, false);
  net_.setInput(blob);

  cv::Mat out = net_.forward(); // expect 1x1xHxW, float
  cv::Mat depth;
  if (out.dims == 4 && out.size[0] == 1 && out.size[1] == 1) {
    depth = cv::Mat(out.size[2], out.size[3], CV_32F, (void*)out.ptr<float>()).clone();
  } else {
    // Fallback: best effort reshape
    depth = out.reshape(1, configuration_.net_input.height).clone();
  }

  cv::Mat depthSrc = with_letterbox_ ? unletterboxDepth(depth, meta)
                                    : [&]{
                                        cv::Mat r; cv::resize(depth, r, bgr.size(), 0,0, cv::INTER_CUBIC);
                                        return r;
                                      }();

  cv::Mat depth01 = normalize(depthSrc);
  cv::Mat color = colorize(depth01);
  cv::Mat visualization; cv::hconcat(bgr, color, visualization);

  ProcessResult r;
  r.visualization = visualization;
  r.raw_primary = depthSrc; // CV_32F depth map aligned to original frame
  return r;
}
