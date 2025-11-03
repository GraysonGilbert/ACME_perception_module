/**
 * @file main.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Example main file for the image processing library
 * @version 0.1
 * @date 2025-10-30
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "analyzer.hpp"
#include <cmath>

int main(int argc, char **argv) {
  const std::string video_path = "../demo_samples/sample.mp4";

  // Use Analyzer to load video and display annotated results
  Analyzer analyzer;
  if (!analyzer.load_video(video_path)) {
    std::cerr << "Failed to open video: " << video_path << "\n";
    return 3;
  }

  bool stop = false;
  while (!stop) {
    std::vector<std::pair<float, float>> results = analyzer.analyze_frame();
    if (results.empty()) {
      std::cout << "No more frames to process or failed to read frame.\n";
      stop = true;
    }

    analyzer.display_frame();

    std::cout << "Press any key in the image window to exit...\n";
    cv::waitKey(0);
  }

  return 0;
}