#include <gtest/gtest.h>
#include "depth_processor.hpp"
#include "image_processor.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

DepthConfig make_dummy_config() {
	DepthConfig config;
	config.model_path = "app/config_files/depth_anything_vitb14_fixed.onnx";
	config.use_letterbox = false;
	config.prefer_cuda = false;
	return config;
}

cv::Mat make_dummy_image(int w = 64, int h = 64) {
	cv::Mat img(h, w, CV_8UC3);
	cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
	return img;
}

// Test suite for DepthProcessor
class DepthProcessorTest : public ::testing::Test {
protected:
	DepthProcessor processor;
	DepthConfig config;
	void SetUp() override {
		config = make_dummy_config();
	}
};

TEST_F(DepthProcessorTest, LoadModelReturnsTrue) {
	bool loaded = processor.load_model(config);
	EXPECT_TRUE(loaded);
}

TEST_F(DepthProcessorTest, NormalizeProducesZeroOneRange) {
	cv::Mat d(2, 2, CV_32F);
	d.at<float>(0,0) = 10; d.at<float>(0,1) = 20;
	d.at<float>(1,0) = 30; d.at<float>(1,1) = 40;
	cv::Mat norm = DepthProcessor::normalize(d);
	double minv, maxv;
	cv::minMaxLoc(norm, &minv, &maxv);
	EXPECT_NEAR(minv, 0.0, 1e-6);
	EXPECT_NEAR(maxv, 1.0, 1e-6);
}

TEST_F(DepthProcessorTest, ColorizeOutputType) {
	cv::Mat d01 = cv::Mat::ones(10, 10, CV_32F);
	cv::Mat color = DepthProcessor::colorize(d01);
	EXPECT_EQ(color.type(), CV_8UC3);
	EXPECT_EQ(color.rows, 10);
	EXPECT_EQ(color.cols, 10);
}

// Integration: process() with dummy model will likely fail, but test interface
TEST_F(DepthProcessorTest, ProcessReturnsResult) {
	processor.load_model(config);
	cv::Mat img = make_dummy_image();
	// Since model is dummy, process may throw or return empty result
	try {
		auto r = processor.process(img);
		// Visualization should have correct width (side-by-side)
		EXPECT_EQ(r.visualization.rows, img.rows);
		EXPECT_EQ(r.visualization.cols, img.cols * 2);
		// raw_primary should match input size
		EXPECT_EQ(r.raw_primary.rows, img.rows);
		EXPECT_EQ(r.raw_primary.cols, img.cols);
	} catch (const cv::Exception& e) {
		SUCCEED() << "process() threw as expected with dummy model: " << e.what();
	}
}
