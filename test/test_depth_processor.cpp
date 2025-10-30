/**
 * @file test_depth_processor.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Unit tests for DepthProcessor class
 * @version 0.1
 * @date 2025-10-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "depth_processor.hpp"

class DepthProcessorTester : public DepthProcessor {
 public:
  using DepthProcessor::colorize;
  using DepthProcessor::normalize;
};

TEST(DepthProcessorTest, LoadModelInvalidPathReturnsFalse) {
  DepthProcessor proc;
  DepthConfig cfg;
  cfg.model_path = "nonexistent_depth_model.onnx";

  bool ok = proc.load_model(cfg);
  EXPECT_FALSE(ok) << "load_model should return false with invalid model path";
}

TEST(DepthProcessorTest, ProcessWithoutModelReturnsEmpty) {
  DepthProcessor proc;

  cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
  DepthResult res = proc.process(img);

  // Current stub implementation should return empty mats.
  EXPECT_TRUE(res.visualization.empty());
  EXPECT_TRUE(res.depth_map.empty());
}

TEST(DepthProcessorTest, ProcessEmptyImageReturnsEmpty) {
  DepthProcessor proc;
  cv::Mat empty;
  DepthResult res = proc.process(empty);

  EXPECT_TRUE(res.visualization.empty());
  EXPECT_TRUE(res.depth_map.empty());
}

TEST(DepthProcessorTest, NormalizeAndColorizeEmptyReturnEmpty) {
  DepthProcessorTester tester;
  cv::Mat empty;

  cv::Mat n = tester.normalize(empty);
  EXPECT_TRUE(n.empty());

  cv::Mat c = tester.colorize(empty);
  EXPECT_TRUE(c.empty());
}

TEST(DepthProcessorTest, LoadModelEmptyModelPathReturnsFalse) {
  DepthProcessor proc;
  DepthConfig cfg;
  cfg.model_path = "";  // empty path should be rejected

  bool ok = proc.load_model(cfg);
  EXPECT_FALSE(ok) << "load_model should return false when model path is empty";
}

TEST(DepthProcessorTest, LoadModelIdeal) {
  DepthProcessor proc;
  DepthConfig cfg;
  cfg.model_path =
      std::string("app/config_files/depth_anything_vitb14_fixed.onnx");

  bool ok = proc.load_model(cfg);
  EXPECT_TRUE(ok) << "load_model should return true for a valid depth model";
}

TEST(DepthProcessorTest, ProcessAfterLoadModelReturnsDepthMaps) {
  DepthProcessor proc;
  DepthConfig cfg;
  cfg.model_path =
      std::string("app/config_files/depth_anything_vitb14_fixed.onnx");

  cv::Mat img = cv::Mat::ones(480, 640, CV_8UC3) * 128;
  DepthResult res = proc.process(img);

  // Expect a visualization (3-channel image) and a floating-point depth map.
  EXPECT_FALSE(res.visualization.empty());
  EXPECT_FALSE(res.depth_map.empty());
  EXPECT_EQ(res.visualization.type() % 8, CV_8U)
      << "visualization should be 8-bit";
  EXPECT_EQ(res.visualization.channels(), 3);
  EXPECT_EQ(res.depth_map.type(), CV_32F);
}

TEST(DepthProcessorTest, NormalizeSimpleRange) {
  DepthProcessorTester tester;

  // Create a simple depth map with known min/max
  cv::Mat d(2, 2, CV_32F);
  d.at<float>(0, 0) = 2.0f;
  d.at<float>(0, 1) = 4.0f;
  d.at<float>(1, 0) = 3.0f;
  d.at<float>(1, 1) = 6.0f;

  cv::Mat n = tester.normalize(d);

  // Expect values in [0,1]
  ASSERT_FALSE(n.empty());
  for (int r = 0; r < n.rows; ++r) {
    for (int c = 0; c < n.cols; ++c) {
      float v = n.at<float>(r, c);
      EXPECT_GE(v, 0.0f);
      EXPECT_LE(v, 1.0f);
    }
  }
}

TEST(DepthProcessorTest, ColorizeProduces3ChannelImage) {
  DepthProcessorTester tester;

  // Create a normalized map (0-1) and colorize it
  cv::Mat norm(4, 4, CV_32F, cv::Scalar(0.5f));
  cv::Mat color = tester.colorize(norm);

  ASSERT_FALSE(color.empty());
  EXPECT_EQ(color.channels(), 3);
  EXPECT_EQ(color.type() % 8, CV_8U)
      << "colorized output should be 8-bit image";
}
