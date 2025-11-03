/**
 * @file test_yolo_processor.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Unit tests for YoloProcessor class
 * @version 0.1
 * @date 2025-10-29
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <opencv2/imgproc.hpp>

#include "yolo_processor.hpp"

using ::testing::Test;

static std::string repo_path_root() {
  return std::string("../app/config_files/");
}

TEST(YoloProcessorTest, LoadModelInvalidModelPathReturnsFalse) {
  YoloProcessor proc;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "classes.txt";  // valid classes
  cfg.model_path =
      repo_path_root() + "nonexistent_model.onnx";  // invalid model path

  bool ok = proc.load_model(cfg);
  EXPECT_FALSE(ok) << "load_model should return false with invalid model path";
}

TEST(YoloProcessorTest, LoadModelInvalidClassesPathReturnsFalse) {
  YoloProcessor proc;
  YoloConfig cfg;
  cfg.classes_path =
      repo_path_root() + "nonexistent_classes.txt";    // invalid classes
  cfg.model_path = repo_path_root() + "yolov5s.onnx";  // valid model

  bool ok = proc.load_model(cfg);
  EXPECT_FALSE(ok)
      << "load_model should return false with invalid classes path";
}

TEST(YoloProcessorTest, LoadModelBothPathsInvalidReturnsFalse) {
  YoloProcessor proc;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "nope_classes.txt";
  cfg.model_path = repo_path_root() + "nope_model.onnx";

  bool ok = proc.load_model(cfg);
  EXPECT_FALSE(ok) << "load_model should return false when both model and "
                      "classes paths are invalid";
}

TEST(YoloProcessorTest, LoadModelEmptyClassesFileReturnsFalse) {
  YoloProcessor proc;
  YoloConfig cfg;
  cfg.model_path = repo_path_root() + "yolov5s.onnx";  // valid model

  // Create a temporary empty classes file and point config at it.
  const std::string tmp = "/tmp/test_empty_classes.txt";
  {
    std::ofstream ofs(tmp, std::ios::trunc);
    // leave file empty
  }
  cfg.classes_path = tmp;

  bool ok = proc.load_model(cfg);
  EXPECT_FALSE(ok) << "load_model should return false when classes file is "
                      "empty (assumption)";

  std::remove(tmp.c_str());
}

TEST(YoloProcessorTest, LoadModelIdeal) {
  YoloProcessor proc;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "classes.txt";
  cfg.model_path = repo_path_root() + "yolov5s.onnx";

  // Expect load_model to succeed with valid configuration
  bool ok = proc.load_model(cfg);
  EXPECT_TRUE(ok)
      << "load_model should return true for valid model and classes";
}

TEST(YoloProcessorTest, ProcessWithoutModelReturnsEmpty) {
  YoloProcessor proc;

  // Do not call load_model. Call process on a simple black image.
  cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
  YoloResult res = proc.process(img);

  // Since model was not loaded (and process is a stub), we expect no
  // detections (empty vectors) and no crash.
  EXPECT_TRUE(res.boxes.empty());
  EXPECT_TRUE(res.class_ids.empty());
  EXPECT_TRUE(res.confidences.empty());
}

TEST(YoloProcessorTest, ProcessEmptyImageReturnsEmpty) {
  YoloProcessor proc;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "classes.txt";
  cfg.model_path = repo_path_root() + "yolov5s.onnx";
  proc.load_model(cfg);

  // Empty cv::Mat
  cv::Mat empty;
  YoloResult res = proc.process(empty);

  // Expect no detections and no crashes.
  EXPECT_TRUE(res.boxes.empty());
  EXPECT_TRUE(res.class_ids.empty());
  EXPECT_TRUE(res.confidences.empty());
}

TEST(YoloProcessorTest, ProcessImage) {
  YoloProcessor proc;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "classes.txt";
  cfg.model_path = repo_path_root() + "yolov5s.onnx";
  proc.load_model(cfg);

  // Small image, different aspect ratio
  cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC3);
  YoloResult res = proc.process(img);

  SUCCEED() << "process completed without crashing on small image.";
}

class YoloProcessorTester : public YoloProcessor {
 public:
  using YoloProcessor::decode;  // make protected method public for tests
};

TEST(YoloProcessorTest, DecodeEmptyOutputReturnsEmpty) {
  YoloProcessorTester tester;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "classes.txt";
  cfg.model_path = repo_path_root() + "yolov5s.onnx";
  tester.load_model(cfg);

  cv::Mat out;  // empty
  std::vector<int> class_ids;
  std::vector<float> confidences;

  auto boxes = tester.decode(out, 0.25f, class_ids, confidences);

  EXPECT_TRUE(boxes.empty());
  EXPECT_TRUE(class_ids.empty());
  EXPECT_TRUE(confidences.empty());
}

TEST(YoloProcessorTest, DecodeSingleRowProducesOneDetection) {
  YoloProcessorTester tester;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "classes.txt";
  cfg.model_path = repo_path_root() + "yolov5s.onnx";
  tester.load_model(cfg);

  // Construct a fake YOLO output row: [cx, cy, w, h, obj_conf, class0_score]
  cv::Mat out(1, 6, CV_32F);
  float cx = 100.0f;
  float cy = 100.0f;
  float w = 50.0f;
  float h = 50.0f;
  float obj = 0.9f;
  float cls0 = 0.8f;
  out.at<float>(0, 0) = cx;
  out.at<float>(0, 1) = cy;
  out.at<float>(0, 2) = w;
  out.at<float>(0, 3) = h;
  out.at<float>(0, 4) = obj;
  out.at<float>(0, 5) = cls0;

  std::vector<int> class_ids;
  std::vector<float> confidences;

  auto boxes = tester.decode(out, 0.25f, class_ids, confidences);

  // Expected: one detection with class 0 and confidence ~ obj * cls0
  ASSERT_EQ(boxes.size(), 1u);
  ASSERT_EQ(class_ids.size(), 1u);
  ASSERT_EQ(confidences.size(), 1u);

  EXPECT_EQ(class_ids[0], 0);
  EXPECT_GE(confidences[0], 0.0f);
  EXPECT_LE(confidences[0], 1.0f);
  float conf_obj = obj;
  float conf_prod = obj * cls0;
  EXPECT_TRUE(std::fabs(confidences[0] - conf_obj) < 1e-4f ||
              std::fabs(confidences[0] - conf_prod) < 1e-4f)
      << "confidence should equal objectness or objectness*class_score";

  // Box should be converted from (cx,cy,w,h) to Rect2f(x,y,w,h) where
  // x = cx - w/2, y = cy - h/2
  cv::Rect2f expected_box(cx - w / 2.0f, cy - h / 2.0f, w, h);
  EXPECT_FLOAT_EQ(boxes[0].x, expected_box.x);
  EXPECT_FLOAT_EQ(boxes[0].y, expected_box.y);
  EXPECT_FLOAT_EQ(boxes[0].width, expected_box.width);
  EXPECT_FLOAT_EQ(boxes[0].height, expected_box.height);
}

TEST(YoloProcessorTest, DecodeMultipleRowsDoesNotCrashAndReturnsMultiple) {
  YoloProcessorTester tester;
  YoloConfig cfg;
  cfg.classes_path = repo_path_root() + "classes.txt";
  cfg.model_path = repo_path_root() + "yolov5s.onnx";
  tester.load_model(cfg);

  // Create 3 identical rows
  cv::Mat out(3, 6, CV_32F);
  for (int r = 0; r < out.rows; ++r) {
    out.at<float>(r, 0) = 100.0f;  // cx
    out.at<float>(r, 1) = 100.0f;  // cy
    out.at<float>(r, 2) = 50.0f;   // w
    out.at<float>(r, 3) = 50.0f;   // h
    out.at<float>(r, 4) = 0.9f;    // obj
    out.at<float>(r, 5) = 0.8f;    // cls0
  }

  std::vector<int> class_ids;
  std::vector<float> confidences;

  auto boxes = tester.decode(out, 0.25f, class_ids, confidences);

  // Expect up to 3 detections (NMS may reduce this).
  // At minimum this should not crash and the output vectors should be
  // internally consistent (same length) and no larger than input rows.
  EXPECT_EQ(class_ids.size(), confidences.size());
  EXPECT_EQ(class_ids.size(), boxes.size());
  EXPECT_LE(boxes.size(), static_cast<size_t>(out.rows));
}
