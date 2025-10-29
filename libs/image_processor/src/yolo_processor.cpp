// YoloProcessor.cpp
#include "yolo_processor.hpp"
#include "image_processor.hpp"
#include <fstream>
#include <iostream>

static std::vector<std::string> loadClasses(const std::string& path) {
  std::vector<std::string> names;
  if (path.empty()) return names;
  std::ifstream ifs(path);
  std::string line;
  while (std::getline(ifs, line)) if (!line.empty()) names.push_back(line);
  return names;
}

bool YoloProcessor::load_model(const CommonConfiguration& baseconfiguration) {
  // Use dynamic_cast if you prefer, but static_cast is fine here by convention
  configuration_ = static_cast<const YoloConfig&>(baseconfiguration);

  // Load class names if not already provided
  if (configuration_.class_names.empty() && !configuration_.classes_path.empty()) {
    configuration_.class_names = loadClasses(configuration_.classes_path);
    if (configuration_.class_names.empty()) {
      std::cerr << "[YOLO] Warning: failed to load classes from " << configuration_.classes_path
                << " — labels will use numeric ids.\n";
    }
  }

  with_letterbox_ = configuration_.use_letterbox;
  input_size_ = configuration_.net_input;

  net_ = cv::dnn::readNet(configuration_.model_path);
  set_backend(net_, configuration_.prefer_cuda);
  output_names_ = net_.getUnconnectedOutLayersNames();
  return !output_names_.empty();
}

static inline std::vector<std::string> getOutputsNames(const cv::dnn::Net& net) {
  std::vector<std::string> names = net.getUnconnectedOutLayersNames();
  return names;
}

std::vector<cv::Rect2f> YoloProcessor::decode(const cv::Mat& out, float confidence_threshold,
                                              std::vector<int>& class_ids,
                                              std::vector<float>& confidences,
                                              const LetterboxMeta* meta) {
  // This decoder layout is YOLOv5/8-style for OpenCV DNN output (Nx85 or Nx(4+1+numClasses)).
  // Adapt indices as needed for your trained graph.
  std::vector<cv::Rect2f> boxes;

  const int numDet = out.size[1];
  const int elem   = out.size[2];

  cv::Mat det(out.size[1], out.size[2], CV_32F, (void*)out.ptr<float>());

  for (int i = 0; i < numDet; ++i) {
    float* p = det.ptr<float>(i);
    float cx = p[0], cy = p[1], w = p[2], h = p[3];
    float boxConf = p[4];

    int bestClass = -1;
    float bestScore = 0.f;
    for (int c = 5; c < elem; ++c) {
      float sc = p[c];
      if (sc > bestScore) { bestScore = sc; bestClass = c-5; }
    }
    float conf = boxConf * bestScore;
    if (conf >= confidence_threshold) {
      // xywh to xy
      float x = cx - w/2.f;
      float y = cy - h/2.f;
      cv::Rect2f b(x,y,w,h);
      if (meta) b = unletterboxBox(b, *meta);
      boxes.push_back(b);
      class_ids.push_back(bestClass);
      confidences.push_back(conf);
    }
  }

  // NMS in original-image coordinates
  std::vector<int> keep;
  std::vector<cv::Rect> intBoxes;
  intBoxes.reserve(boxes.size());
  for (auto& b : boxes) intBoxes.emplace_back(b);
  cv::dnn::NMSBoxes(intBoxes, confidences, configuration_.confidence_threshold, configuration_.nms_threshold, keep);

  std::vector<cv::Rect2f> outBoxes;
  std::vector<int> outIds;
  std::vector<float> outconfidences;
  for (int k : keep) {
    outBoxes.push_back(boxes[k]);
    outIds.push_back(class_ids[k]);
    outconfidences.push_back(confidences[k]);
  }
  boxes.swap(outBoxes);
  class_ids.swap(outIds);
  confidences.swap(outconfidences);
  return boxes;
}

cv::Mat YoloProcessor::draw(const cv::Mat& frame,
                            const std::vector<cv::Rect2f>& boxes,
                            const std::vector<int>& class_ids,
                            const std::vector<float>& confidences,
                            const std::vector<std::string>& names) {
  cv::Mat visualization = frame.clone();
  const bool haveNames = !names.empty();
  for (size_t i = 0; i < boxes.size(); ++i) {
    cv::rectangle(visualization, boxes[i], {0,255,0}, 2);

    int id = (i < class_ids.size() ? class_ids[i] : -1);
    std::string cname = "id=" + std::to_string(std::max(id, 0));
    if (haveNames && id >= 0 && id < (int)names.size()) cname = names[id];

    char label[128];
    std::snprintf(label, sizeof(label), "%s: %.2f", cname.c_str(), confidences[i]);
    cv::putText(visualization, label, boxes[i].tl() + cv::Point2f(0.f, -4.f),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 1);
  }
  return visualization;
}

ProcessResult YoloProcessor::process(const cv::Mat& bgr) {
  cv::Mat input;
  LetterboxMeta meta;
  if (with_letterbox_) {
    meta = letterbox(bgr, input, input_size_, {114,114,114});
  } else {
    cv::resize(bgr, input, input_size_);
  }

  cv::Mat blob = cv::dnn::blobFromImage(
      input, configuration_.scale, input_size_, configuration_.mean, configuration_.swap_RB, false);

  net_.setInput(blob);

  std::vector<cv::Mat> outs;
  net_.forward(outs, output_names_); // typically single output for YOLOv5/8 after export
  cv::Mat out = outs.size() ? outs[0] : net_.forward();

  std::vector<int> class_ids; std::vector<float> confidences;
  auto boxes = decode(out, configuration_.confidence_threshold, class_ids, confidences,
                      with_letterbox_ ? &meta : nullptr);

  cv::Mat visualization = draw(bgr, boxes, class_ids, confidences, configuration_.class_names);

  YoloResult result;
  result.visualization = visualization;
  result.raw_primary = out; // raw detection tensor if you want it
  result.boxes = boxes;
  result.class_ids = class_ids;
  result.confidences = confidences;
  return result;
}
