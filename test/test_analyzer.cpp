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

#include <cstdio>
#include <fstream>

#include "analyzer.hpp"
#include <sys/stat.h>

// Simple helper to test file existence using std::ifstream
static bool file_exists(const std::string& path) {
  std::ifstream ifs(path);
  return ifs.good();
}

TEST(AnalyzerTest, ConstructDoesNotThrow) { EXPECT_NO_THROW(Analyzer a;); }

TEST(AnalyzerTest, LoadVideoInvalidPathReturnsFalse) {
  Analyzer a;
  bool ok = a.load_video("/nonexistent/path/to/video.mp4");
  EXPECT_FALSE(ok) << "load_video should return false for an invalid path";
}


TEST(AnalyzerTest, LoadVideo) {
  Analyzer a;
  bool ok = a.load_video("../demo_samples/sample.mp4");
  EXPECT_TRUE(ok) << "load_video should return true for a valid path";
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
  EXPECT_TRUE(pts.empty()) << "analyze_frame should return empty vector when "
                              "no video/camera is loaded";
}

TEST(AnalyzerTest, PrintAnalysisToFileCreatesFile) {
  Analyzer a;
  const std::string tmp = "/tmp/test_analyzer_output.txt";

  // remove file if it exists
  std::remove(tmp.c_str());

  a.print_analysis_to_file(tmp);

  EXPECT_TRUE(file_exists(tmp))
      << "print_analysis_to_file should create the output file";

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
    cv::cvtColor((cv::Mat)(d.depth_map * 127.0f / 2.0f), vis,
                 cv::COLOR_GRAY2BGR);
    vis.convertTo(d.visualization, CV_8UC3);
    return d;
  }
};

class TestableAnalyzer : public Analyzer {
 public:
  using Analyzer::set_depth_processor_for_test;
  using Analyzer::set_frame_for_test;
  using Analyzer::set_yolo_processor_for_test;
  using Analyzer::get_yolo_classes_for_test;
};

// Helper to read an entire file into a string for content assertions
static std::string read_file_to_string(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) return std::string();
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

// Fake YOLO that reports a non-person class (class id -1) so Analyzer should
// ignore it (Analyzer only reports positions for "person" detections).
class FakeYoloNonPerson : public YoloProcessor {
 public:
  YoloResult process(const cv::Mat& bgr) {
    YoloResult r;
    r.boxes.push_back(cv::Rect2f(10.0f, 10.0f, 20.0f, 20.0f));
    r.class_ids.push_back(-1);
    r.confidences.push_back(0.50f);
    return r;
  }
};

TEST(AnalyzerTest, NonPersonDetectionsAreIgnored) {
  TestableAnalyzer a;

  a.set_yolo_processor_for_test(std::make_unique<FakeYoloNonPerson>());
  a.set_depth_processor_for_test(std::make_unique<FakeDepth>());

  cv::Mat frame = cv::Mat::zeros(120, 160, CV_8UC3);
  a.set_frame_for_test(frame);

  auto pts = a.analyze_frame();
  EXPECT_TRUE(pts.empty()) << "Non-person detections should not produce positions";
}

TEST(AnalyzerTest, PersonDetection) {
  TestableAnalyzer a;

  a.set_yolo_processor_for_test(std::make_unique<FakeYolo>());
  a.set_depth_processor_for_test(std::make_unique<FakeDepth>());

  cv::Mat frame = cv::Mat::zeros(120, 160, CV_8UC3);
  a.set_frame_for_test(frame);

  auto pts = a.analyze_frame();
  EXPECT_TRUE(!pts.empty()) << "Person detections should produce positions";
}

TEST(AnalyzerTest, YoloClassesNotEmptyAfterConstruct) {
  TestableAnalyzer a;

  const auto& classes = a.get_yolo_classes_for_test();
  EXPECT_FALSE(classes.empty()) << "YOLO classes should be loaded and not empty";
}


