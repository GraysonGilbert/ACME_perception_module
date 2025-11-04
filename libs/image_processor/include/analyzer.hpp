/**
 * @file analyzer.hpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief The anlyzer class processes images frame by frame and provides a count
 * of humans and their locations.
 * @version 0.1
 * @date 2025-10-23
 *
 * @copyright Copyright (c) 2025
 *
 */

/*
 * MIT License
 * 
 * Copyright (c) 2025 Grayson G. & Marcus Hurt
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "depth_processor.hpp"
#include "yolo_processor.hpp"

class Analyzer {
 public:
  Analyzer();

  /**
   * @brief Analyzes a single frame from the video source.
   *
   * @return std::vector<std::pair<float, float>> - A vector of (x, y)
   * coordinates of detected humans in the frame with respect to the camera
   * origin.
   */
  std::vector<std::pair<float, float>> analyze_frame();

  /**
   * @brief print the last analyzed frame to the console.
   *
   */
  void display_frame();

  /**
   * @brief prints the analysis of the last frame to a file.
   *
   * @param file_path - string of the absolute file path to write to.
   */
  void print_analysis_to_file(const std::string& file_path);

  /**
   * @brief loads a video from a file path to be analyzed.
   *
   * @param video_path - string of the absolute file path to the video.
   * @return true - successfully loaded video.
   * @return false - failed to load video.
   */
  bool load_video(const std::string& video_path);

  /**
   * @brief loads a camera by its index to be analyzed.
   *
   * @param camera_index - index of the camera to load (default is 0).
   * @return true - successfully loaded camera.
   * @return false - failed to load camera.
   */
  bool load_camera(int camera_index = 0);

 protected:
  void set_yolo_processor_for_test(std::unique_ptr<YoloProcessor> p);
  void set_depth_processor_for_test(std::unique_ptr<DepthProcessor> p);
  void set_frame_for_test(const cv::Mat& f);
  std::vector<std::string> get_yolo_classes_for_test() const;

 private:
  cv::VideoCapture capture_;
  cv::Mat frame_;
  cv::Mat annotated_frame_;

  YoloConfig yolo_configuration_;
  std::unique_ptr<YoloProcessor> yolo_processor_;
  YoloResult yolo_result_;
  std::vector<std::string> yolo_classes_;

  DepthConfig depth_configuration_;
  std::unique_ptr<DepthProcessor> depth_processor_;
  DepthResult depth_result_;

  const std::string DEPTH_MODEL_PATH =
      "config_files/depth_anything_vitb14_fixed.onnx";
  const std::string YOLO_MODEL_PATH = "config_files/yolov5s.onnx";
  const std::string YOLO_CLASSES_PATH = "config_files/classes.txt";
};
