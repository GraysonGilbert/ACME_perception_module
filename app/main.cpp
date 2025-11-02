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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolo_processor.hpp"
#include "depth_processor.hpp"
#include <cmath>

static std::vector<std::string> load_classes(const std::string &path) {
  std::vector<std::string> classes;
  std::ifstream ifs(path);
  if (!ifs) return classes;
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty()) classes.push_back(line);
  }
  return classes;
}

int main(int argc, char **argv) {
  // Paths relative to repository root
  const std::string classes_path = "app/config_files/classes.txt";
  const std::string model_path = "app/config_files/yolov5s.onnx";
  const std::string video_path = "local/depth-opencv-cpp/sample.mp4";

  YoloConfig cfg;
  cfg.classes_path = classes_path;
  cfg.model_path = model_path;

  YoloProcessor proc;
  std::cout << "Loading YOLO model...\n";
  if (!proc.load_model(cfg)) {
    std::cerr << "Failed to load YOLO model or classes (" << model_path << ", "
              << classes_path << ")\n";
    return 2;
  }

  // Also load class names locally so we can print names
  auto classes = load_classes(classes_path);

  // Load depth model (optional - warn if it fails)
  DepthProcessor dproc;
  DepthConfig dcfg;
  dcfg.model_path = "app/config_files/depth_anything_vitb14_fixed.onnx";
  bool depth_ok = dproc.load_model(dcfg);
  if (!depth_ok) {
    std::cerr << "Warning: failed to load depth model (" << dcfg.model_path
              << ") - depth visualization/estimates will be unavailable\n";
  }

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video: " << video_path << "\n";
    return 3;
  }

  cv::Mat frame;
  if (!cap.read(frame) || frame.empty()) {
    std::cerr << "Failed to read first frame from video\n";
    return 4;
  }

  std::cout << "Running YOLO on first frame...\n";
  YoloResult res = proc.process(frame);

  // Also run depth processing if model loaded
  DepthResult dres;
  if (depth_ok) dres = dproc.process(frame);

  std::cout << "Detections: " << res.boxes.size() << "\n";
  for (size_t i = 0; i < res.boxes.size(); ++i) {
    const auto &b = res.boxes[i];
    int cls = (i < res.class_ids.size()) ? res.class_ids[i] : -1;
    float conf = (i < res.confidences.size()) ? res.confidences[i] : 0.0f;
    std::string name = (cls >= 0 && static_cast<size_t>(cls) < classes.size())
                           ? classes[cls]
                           : std::to_string(cls);
    std::cout << i << ": class=" << name << " id=" << cls << " conf=" << conf
              << " box=[" << b.x << ", " << b.y << ", " << b.width << ", "
              << b.height << "]\n";
  }

  // Draw detections on the frame
  // Define a small palette
  const std::vector<cv::Scalar> colors = {
      cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
      cv::Scalar(0, 255, 0),   cv::Scalar(0, 128, 255), cv::Scalar(255, 0, 0)};

  cv::Mat annotated = frame.clone();
  for (size_t i = 0; i < res.boxes.size(); ++i) {
    const auto &b = res.boxes[i];
    int cls = (i < res.class_ids.size()) ? res.class_ids[i] : -1;
    float conf = (i < res.confidences.size()) ? res.confidences[i] : 0.0f;

    cv::Scalar color =
        colors[cls >= 0 ? (cls % colors.size()) : (i % colors.size())];

    // Convert float rect to int rect and clamp
    int x = std::max(0, static_cast<int>(std::round(b.x)));
    int y = std::max(0, static_cast<int>(std::round(b.y)));
    int w_box = std::max(0, static_cast<int>(std::round(b.width)));
    int h_box = std::max(0, static_cast<int>(std::round(b.height)));
    if (x + w_box > annotated.cols) w_box = annotated.cols - x;
    if (y + h_box > annotated.rows) h_box = annotated.rows - y;

    cv::Rect box(x, y, w_box, h_box);
    cv::rectangle(annotated, box, color, 2);

    // compute center and optionally sample depth
    int cx_pixel = x + w_box / 2;
    int cy_pixel = y + h_box / 2;

    // position string (empty if depth not available)
    std::string posstr;
    bool has_pos = false;

    // If we have a valid depth map, sample the depth at the center
    if (!dres.depth_map.empty() && cy_pixel >= 0 && cy_pixel < dres.depth_map.rows &&
        cx_pixel >= 0 && cx_pixel < dres.depth_map.cols) {
      float Z = dres.depth_map.at<float>(cy_pixel, cx_pixel);
      if (std::isfinite(Z) && Z > 0.0f) {
        // Assume a pinhole camera with a horizontal FOV of 60 degrees
        const double hfov_deg = 60.0;
        const double hfov_rad = hfov_deg * M_PI / 180.0;
        const double fx = (frame.cols / 2.0) / std::tan(hfov_rad / 2.0);
        const double fy = fx;  // assume square pixels
        const double cx_cam = frame.cols / 2.0;
        const double cy_cam = frame.rows / 2.0;

        double X = (static_cast<double>(cx_pixel) - cx_cam) * Z / fx;
        double Y = (static_cast<double>(cy_pixel) - cy_cam) * Z / fy;

        std::ostringstream posss;
        posss << "(X=" << std::fixed << std::setprecision(2) << X << " m, Y="
              << Y << " m, Z=" << Z << " m)";
        posstr = posss.str();
        has_pos = true;

        // Draw center point
        cv::circle(annotated, cv::Point(cx_pixel, cy_pixel), 3, cv::Scalar(0, 0, 0), -1);

        // Print to stdout for each person
        std::string name = (i < res.class_ids.size() && res.class_ids[i] >= 0 &&
                            static_cast<size_t>(res.class_ids[i]) < classes.size())
                               ? classes[res.class_ids[i]]
                               : std::to_string((i < res.class_ids.size()) ? res.class_ids[i] : -1);
        if (name == "person") {
          std::cout << "Person " << i << " center pixel (" << cx_pixel << ","
                    << cy_pixel << ") -> position " << posstr << "\n";
        }
      }
    }

    // Prepare class/conf label text
    std::ostringstream ss;
    if (cls >= 0 && static_cast<size_t>(cls) < classes.size())
      ss << classes[cls];
    else
      ss << "cls=" << cls;
    ss << ": " << std::fixed << std::setprecision(2) << conf;
    std::string label = ss.str();

    double fontScale = 0.5;
    int thickness = 1;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;

    int baseLine1 = 0;
    cv::Size size1 = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine1);
    int baseLine2 = 0;
    cv::Size size2(0,0);
    if (has_pos) size2 = cv::getTextSize(posstr, fontFace, fontScale*0.9, thickness, &baseLine2);

    int pad = 6;
    int spacing = 4;
    int total_width = std::max(size1.width, size2.width) + pad * 2;
    int total_height = size1.height + (has_pos ? (spacing + size2.height) : 0) + pad * 2;

    int tl_x = x;
    if (tl_x + total_width > annotated.cols) tl_x = std::max(0, annotated.cols - total_width - 2);

    // Prefer to place the label above the box; if not enough room, place below
    int tl_y_above = y - total_height - 2;
    int tl_y = (tl_y_above >= 0) ? tl_y_above : (y + h_box + 2);
    if (tl_y + total_height > annotated.rows) tl_y = std::max(0, annotated.rows - total_height - 2);

    cv::Point label_tl(tl_x, tl_y);
    cv::Point label_br(tl_x + total_width, tl_y + total_height);

    // Draw filled rectangle for label background (use same color)
    cv::rectangle(annotated, label_tl, label_br, color, cv::FILLED);

    // Put text lines in black for contrast
    int y_text1 = tl_y + pad + size1.height;
    cv::putText(annotated, label, cv::Point(tl_x + pad, y_text1), fontFace,
                fontScale, cv::Scalar(0, 0, 0), thickness);
    if (has_pos) {
      int y_text2 = y_text1 + spacing + size2.height;
      cv::putText(annotated, posstr, cv::Point(tl_x + pad, y_text2), fontFace,
                  fontScale*0.9, cv::Scalar(0, 0, 0), thickness);
    }
  }

  // Show annotated image and depth visualization side-by-side when available
  const std::string win = "YOLO Detections";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
  if (!dres.visualization.empty()) {
    cv::Mat depth_vis = dres.visualization;
    // ensure same height as annotated (resize preserving aspect)
    if (depth_vis.rows != annotated.rows || depth_vis.cols != annotated.cols) {
      cv::resize(depth_vis, depth_vis, annotated.size(), 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat combined;
    cv::hconcat(annotated, depth_vis, combined);
    cv::imshow(win, combined);
  } else {
    cv::imshow(win, annotated);
  }
  std::cout << "Press any key in the image window to exit...\n";
  cv::waitKey(0);

  return 0;
}