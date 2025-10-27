/**
 * @file yolo_processor.cpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief 
 * @version 0.1
 * @date 2025-10-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */


#include "yolo_processor.hpp"
#include <iostream>

bool YoloProcessor::load_model(const CommonConfiguration& configuration) {
    std::cout << "[StubYoloProcessor] load_model() called (stub)" << std::endl;
    configuration_ = static_cast<const YoloConfig&>(configuration);
    // Pretend to load a model successfully
    return true;
}

ProcessResult YoloProcessor::process(const cv::Mat& bgr) {
    std::cout << "[StubYoloProcessor] process() called (stub)" << std::endl;

    // Create dummy YOLO result
    YoloResult result;
    result.visualization = bgr.empty() ? cv::Mat::zeros(100, 100, CV_8UC3) : bgr.clone();
    result.raw_primary = result.visualization.clone();

    // Dummy detections
    result.boxes.push_back(cv::Rect2f(50, 50, 100, 150));
    result.class_ids.push_back(0);
    result.confidences.push_back(0.9f);

    // Draw boxes on the visualization (for realism)
    result.visualization = draw(result.visualization, result.boxes, result.class_ids,
                                result.confidences, configuration_.class_names);

    return result;
}

std::vector<cv::Rect2f> YoloProcessor::decode(const cv::Mat&, float confidence_threshold, std::vector<int>& class_ids,
                                              std::vector<float>& confidences, const LetterboxMeta*) {
    std::cout << "[StubYoloProcessor] decode() called (stub)" << std::endl;

    std::vector<cv::Rect2f> boxes;
    return boxes;
}

cv::Mat YoloProcessor::draw(const cv::Mat& frame, const std::vector<cv::Rect2f>& boxes, const std::vector<int>& class_ids,
                            const std::vector<float>& confidences, const std::vector<std::string>& names) {
    cv::Mat output = frame.clone();
    return output;
}
