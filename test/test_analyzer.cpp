/**
 * @file test_analyzer.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Unit tests for Analyzer class
 * @version 0.1
 * @date 2025-10-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <gtest/gtest.h>

#include <fstream>
#include <cstdio>

#include "analyzer.hpp"

// Simple helper to test file existence using std::ifstream
static bool file_exists(const std::string& path) {
  std::ifstream ifs(path);
  return ifs.good();
}

TEST(AnalyzerTest, ConstructDoesNotThrow) {
  EXPECT_NO_THROW(Analyzer a;);
}

TEST(AnalyzerTest, LoadVideoInvalidPathReturnsFalse) {
  Analyzer a;
  bool ok = a.load_video("/nonexistent/path/to/video.mp4");
  EXPECT_FALSE(ok) << "load_video should return false for an invalid path";
}

TEST(AnalyzerTest, LoadCameraInvalidIndexReturnsFalse) {
  Analyzer a;
  // Negative camera index should be rejected
  bool ok = a.load_camera(-99);
  EXPECT_FALSE(ok) << "load_camera should return false for invalid index";
}

TEST(AnalyzerTest, AnalyzeFrameWithoutLoadReturnsEmpty) {
  Analyzer a;
  auto pts = a.analyze_frame();
  EXPECT_TRUE(pts.empty()) << "analyze_frame should return empty vector when no video/camera is loaded";
}

TEST(AnalyzerTest, PrintAnalysisToFileCreatesFile) {
  Analyzer a;
  const std::string tmp = "/tmp/test_analyzer_output.txt";
  // remove file if it exists
  std::remove(tmp.c_str());

  a.print_analysis_to_file(tmp);

  EXPECT_TRUE(file_exists(tmp)) << "print_analysis_to_file should create the output file";

  std::remove(tmp.c_str());
}

// Fake YoloProcessor that returns a single detection at a known location.
class FakeYolo : public YoloProcessor {
 public:
  YoloResult process(const cv::Mat& bgr) {
    YoloResult r;
    // one box centered at (320,240) size 100x200
    r.boxes.push_back(cv::Rect2f(270.0f, 140.0f, 100.0f, 200.0f));
    r.class_ids.push_back(0);
    r.confidences.push_back(0.95f);
    return r;
  }
};

// Fake DepthProcessor that returns a depth map with a known value at the
// detection center.
class FakeDepth : public DepthProcessor {
 public:
  DepthResult process(const cv::Mat& bgr) {
    DepthResult d;
    // depth_map single-channel float, same size as input
    d.depth_map = cv::Mat::ones(bgr.rows, bgr.cols, CV_32F) * 2.0f;  // 2 meters
    // visualization as simple 3-channel gray image
    cv::Mat vis;
    cv::cvtColor((cv::Mat)(d.depth_map * 127.0f / 2.0f), vis, cv::COLOR_GRAY2BGR);
    vis.convertTo(d.visualization, CV_8UC3);
    return d;
  }
};

class TestableAnalyzer : public Analyzer {
 public:
  using Analyzer::set_yolo_processor_for_test;
  using Analyzer::set_depth_processor_for_test;
  using Analyzer::set_frame_for_test;
};

TEST(AnalyzerTest, AnalyzeFrameWithInjectedProcessorsProducesCoordinates) {
  TestableAnalyzer a;

  // Inject fakes
  a.set_yolo_processor_for_test(std::make_unique<FakeYolo>());
  a.set_depth_processor_for_test(std::make_unique<FakeDepth>());

  cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
  cv::rectangle(frame, cv::Point(270, 140), cv::Point(370, 340), cv::Scalar(255, 255, 255), -1);

  a.set_frame_for_test(frame);

  auto pts = a.analyze_frame();

  ASSERT_EQ(pts.size(), 1u);
  EXPECT_NEAR(pts[0].first, 320.0f, 1e-3f);
  EXPECT_NEAR(pts[0].second, 2.0f, 1e-3f);
}