
/**
 * @file test_letterbox.cpp
 * @brief Tests for letterbox and unletterbox helpers
 */

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "letterbox.hpp"

TEST(LetterboxTest, BasicLetterboxPadsAndScales) {
  // source image 300x200
  cv::Mat src(200, 300, CV_8UC3, cv::Scalar(10, 20, 30));

  // draw a distinctive pixel at (x=10,y=20)
  src.at<cv::Vec3b>(20, 10) = cv::Vec3b(200, 150, 100);

  cv::Size dest_size(640, 640);
  cv::Mat dest;

  LetterboxMeta meta = letterbox(src, dest, dest_size);

  // destination must have requested size
  EXPECT_EQ(dest.cols, dest_size.width);
  EXPECT_EQ(dest.rows, dest_size.height);

  // scale matches computation
  float expected_scale = std::min(dest_size.width / (float)src.cols,
                                  dest_size.height / (float)src.rows);
  EXPECT_NEAR(meta.scale, expected_scale, 1e-6f);

  // compute expected new dims and pads
  int new_w = int(std::round(src.cols * expected_scale));
  int new_h = int(std::round(src.rows * expected_scale));
  int exp_pad_left = (dest_size.width - new_w) / 2;
  int exp_pad_top = (dest_size.height - new_h) / 2;
  EXPECT_EQ(meta.pad_left, exp_pad_left);
  EXPECT_EQ(meta.pad_top, exp_pad_top);

  // pixel in padded area should equal pad value (default 114)
  cv::Scalar pad_val(114, 114, 114);
  if (meta.pad_top > 0) {
    // choose a point inside top padding region
    int py = meta.pad_top / 2;
    int px = meta.pad_left + 1;
    auto pix = dest.at<cv::Vec3b>(py, px);
    EXPECT_EQ(pix[0], static_cast<uchar>(pad_val[0]));
    EXPECT_EQ(pix[1], static_cast<uchar>(pad_val[1]));
    EXPECT_EQ(pix[2], static_cast<uchar>(pad_val[2]));
  }

  // the distinctive source pixel should appear at scaled+pad location
  int src_x = 10;
  int src_y = 20;
  int dest_x = int(std::round(src_x * meta.scale)) + meta.pad_left;
  int dest_y = int(std::round(src_y * meta.scale)) + meta.pad_top;
  int roi_left = meta.pad_left;
  int roi_top = meta.pad_top;
  int roi_w = meta.destination_size.width - (meta.pad_left + meta.pad_right);
  int roi_h = meta.destination_size.height - (meta.pad_top + meta.pad_bottom);
  int center_x_roi = roi_left + roi_w / 2;
  int center_y_roi = roi_top + roi_h / 2;

  cv::Vec3b center_pix = dest.at<cv::Vec3b>(center_y_roi, center_x_roi);
  cv::Vec3b pad_color(static_cast<uchar>(pad_val[0]),
                      static_cast<uchar>(pad_val[1]),
                      static_cast<uchar>(pad_val[2]));
  EXPECT_NE(center_pix, pad_color);

  // Also ensure there exists at least one non-pad pixel inside the ROI
  bool found_non_pad = false;
  for (int y = roi_top; y < roi_top + roi_h && !found_non_pad; ++y) {
    for (int x = roi_left; x < roi_left + roi_w; ++x) {
      if (dest.at<cv::Vec3b>(y, x) != pad_color) {
        found_non_pad = true;
        break;
      }
    }
  }
  EXPECT_TRUE(found_non_pad);
}

TEST(LetterboxTest, UnletterboxBoxAndDepth) {
  // source image 123x77
  cv::Mat src(77, 123, CV_8UC3, cv::Scalar(1, 2, 3));
  cv::Size dest_size(320, 320);
  cv::Mat dest;
  LetterboxMeta meta = letterbox(src, dest, dest_size);

  // original box in source coords
  cv::Rect2f orig_box(10.5f, 5.0f, 50.0f, 20.0f);

  // box in letterboxed (destination) coords
  cv::Rect2f letterboxed_box(orig_box.x * meta.scale + meta.pad_left,
                             orig_box.y * meta.scale + meta.pad_top,
                             orig_box.width * meta.scale,
                             orig_box.height * meta.scale);

  cv::Rect2f restored = unletterboxBox(letterboxed_box, meta);

  EXPECT_NEAR(restored.x, orig_box.x, 1e-3f);
  EXPECT_NEAR(restored.y, orig_box.y, 1e-3f);
  EXPECT_NEAR(restored.width, orig_box.width, 1e-3f);
  EXPECT_NEAR(restored.height, orig_box.height, 1e-3f);

  // test unletterboxDepth: create a destination depth map filled with 42.0f
  cv::Mat depth_dest(dest_size.height, dest_size.width, CV_32F, 42.0f);
  cv::Mat restored_depth = unletterboxDepth(depth_dest, meta);

  // restored_depth should match source size and values
  EXPECT_EQ(restored_depth.cols, src.cols);
  EXPECT_EQ(restored_depth.rows, src.rows);
  for (int r = 0; r < restored_depth.rows; ++r) {
    for (int c = 0; c < restored_depth.cols; ++c) {
      // Allow a small numerical tolerance due to resize interpolation
      EXPECT_NEAR(restored_depth.at<float>(r, c), 42.0f, 1e-5f);
    }
  }
}
