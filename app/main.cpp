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

  int frame_count = 0;
  const int max_frames = 120;
  auto start_time = std::chrono::high_resolution_clock::now();
  while (frame_count < max_frames) {  // process up to 120 frames
    frame_count++;
    std::vector<std::pair<float, float>> results = analyzer.analyze_frame();

    analyzer.display_frame();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double fps = frame_count / elapsed.count();
    std::cout << "Frame " << frame_count << " FPS: " << fps << "\n";
  }

  return 0;
}