/**
 * @file analyzer.hpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief The anlyzer class processes images frame by frame and provides a count of humans and their locations.
 * @version 0.1
 * @date 2025-10-23
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

#include "depth_processor.hpp"
#include "image_processor.hpp"
#include "yolo_processor.hpp"

class Analyzer {
public:
    Analyzer();

    std::vector<std::pair<float, float>> analyze_frame();

    void print_frame_to_console();
    void print_analysis_to_file(const std::string& file_path);
    
    bool load_video(const std::string& video_path);
    bool load_camera(int camera_index = 0);

private:
    cv::VideoCapture capture_;
    cv::Mat frame_;

    YoloConfig yolo_configuration_;
    std::unique_ptr<YoloProcessor> yolo_processor_;
    YoloResult yolo_result_;
    
    DepthConfig depth_configuration_;
    std::unique_ptr<DepthProcessor> depth_processor_;
    ProcessResult depth_result_;

    const std::string DEPTH_MODEL_PATH = "config_files/depth_anything_vitb14_fixed.onnx";
    const std::string YOLO_MODEL_PATH = "config_files/yolov5s.onnx";
    const std::string YOLO_CLASSES_PATH = "config_files/classes.txt";
};

