/**
 * @file letterbox.hpp
 * @author Marcus Hurt (mhurt@umd.edu)
 * @author Grayson Gilbert (ggilbert@umd.edu)
 * @brief Functions for letterboxing images and removing letterboxing from
 * bounding boxes and depth maps
 * @version 0.1
 * @date 2025-10-30
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
#include <opencv2/opencv.hpp>

struct LetterboxMeta {
  cv::Size source_size{};
  cv::Size destination_size{};
  float scale{1.f};
  int pad_top{0}, pad_bottom{0}, pad_left{0}, pad_right{0};
};

/**
 * @brief letterboxes the source image into the destination image of given size.
 *
 * @param src - source cv::Mat image
 * @param destination - output cv::Mat
 * @param destination_size - desired sixe
 * @param pad_value - value to pad the image with
 * @return LetterboxMeta - Leterboxed metadata
 */
inline LetterboxMeta letterbox(const cv::Mat& src, cv::Mat& destination,
                               const cv::Size& destination_size,
                               const cv::Scalar& pad_value = {114, 114, 114}) {
  LetterboxMeta metadata;
  metadata.source_size = src.size();
  metadata.destination_size = destination_size;
  float ratio = std::min(destination_size.width / (float)src.cols,
                         destination_size.height / (float)src.rows);
  metadata.scale = ratio;

  int new_width = int(std::round(src.cols * ratio));
  int new_height = int(std::round(src.rows * ratio));
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_width, new_height), 0, 0,
             cv::INTER_LINEAR);

  int destination_width = destination_size.width - new_width;
  int destination_height = destination_size.height - new_height;
  metadata.pad_left = destination_width / 2;
  metadata.pad_right = destination_width - metadata.pad_left;
  metadata.pad_top = destination_height / 2;
  metadata.pad_bottom = destination_height - metadata.pad_top;

  cv::copyMakeBorder(resized, destination, metadata.pad_top,
                     metadata.pad_bottom, metadata.pad_left, metadata.pad_right,
                     cv::BORDER_CONSTANT, pad_value);
  return metadata;
}

/**
 * @brief Removes letterboxing from a bounding box using the provided metadata.
 *
 * @param box - bounding box in letterboxed coordinates
 * @param metadata - LetterboxMeta used during letterboxing
 * @return cv::Rect2f - bounding box in original image coordinates
 */
inline cv::Rect2f unletterboxBox(const cv::Rect2f& box,
                                 const LetterboxMeta& metadata) {
  // box is in destination (letterboxed) coords; map back to original image
  // coords
  float x = (box.x - metadata.pad_left) / metadata.scale;
  float y = (box.y - metadata.pad_top) / metadata.scale;
  float w = box.width / metadata.scale;
  float h = box.height / metadata.scale;
  return {x, y, w, h};
}

/**
 * @brief Removes letterboxing from a depth map using the provided metadata.
 *
 * @param depth_destination - depth map in letterboxed coordinates
 * @param metadata - LetterboxMeta used during letterboxing
 * @return cv::Mat - depth map in original image coordinates
 */
inline cv::Mat unletterboxDepth(const cv::Mat& depth_destination,
                                const LetterboxMeta& metadata) {
  // depthdestination is HxW (destination_size) CV_32F; remove padding then
  // resize back
  cv::Rect region_of_interest(metadata.pad_left, metadata.pad_top,
                              metadata.destination_size.width -
                                  (metadata.pad_left + metadata.pad_right),
                              metadata.destination_size.height -
                                  (metadata.pad_top + metadata.pad_bottom));
  cv::Mat cropped = depth_destination(region_of_interest);
  cv::Mat restored;
  cv::resize(cropped, restored, metadata.source_size, 0, 0, cv::INTER_CUBIC);
  return restored;
}
