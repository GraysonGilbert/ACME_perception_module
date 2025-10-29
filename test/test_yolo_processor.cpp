#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "yolo_processor.hpp"
#include "image_processor.hpp"

class YoloProcessorTest : public ::testing::Test {
protected:
	YoloProcessor processor;
	YoloConfig configuration;

	void SetUp() override {
		// Set up a minimal config (paths may need to be adjusted for real tests)
		configuration.confidence_threshold = 0.4f;
		configuration.nms_threshold = 0.4f;
		configuration.classes_path = "app/config_files/classes.txt";
		configuration.net_input = cv::Size(640, 640);
        configuration.scale    = 1.0/255.0;
        configuration.swap_RB   = true;
        configuration.mean     = {0,0,0};
        configuration.std      = {1,1,1};
	}
};

TEST_F(YoloProcessorTest, LoadModelFailsWithInvalidPath) {
	configuration.model_path = "nonexistent.txt";
	EXPECT_FALSE(processor.load_model(configuration));
}

TEST_F(YoloProcessorTest, LoadModelSucceedsWithValidPath) {
	configuration.model_path = "app/config_files/yolov5s.onnx";
	bool loaded = processor.load_model(configuration);
    EXPECT_TRUE(loaded);
}

TEST_F(YoloProcessorTest, ProcessReturnsResult) {
    configuration.model_path = "app/config_files/yolov5s.onnx";
    ASSERT_TRUE(processor.load_model(configuration));
    cv::Mat dummy = cv::Mat::ones(640, 640, CV_8UC3);
	ProcessResult result = processor.process(dummy);
	// Check that result fields are valid (should be empty or default)
	EXPECT_TRUE(result.visualization.empty() || result.visualization.type() == CV_8UC3);
	EXPECT_TRUE(result.raw_primary.empty() || result.raw_primary.type() == CV_32F);
}




