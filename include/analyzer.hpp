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

    /**
     * @brief Analyzes a single frame from the video source.
     * 
     * @return std::vector<std::pair<float, float>> - A vector of (x, y) coordinates of detected humans in the frame.
     */
    std::vector<std::pair<float, float>> analyze_frame();

    /**
     * @brief print the last analyzed frame to the console.
     * 
     */
    void print_frame_to_console();

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

