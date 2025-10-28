// main.cpp
#include <iostream>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "image_processor.hpp"
#include "yolo_processor.hpp"
#include "depth_processor.hpp"

// ---------- CLI parsing helpers ----------
static void print_usage(const char* prog) {
  std::cout <<
    "Usage:\n"
    "  " << prog << " <yolo|depth> <model.onnx> [options] [video_path]\n\n"
    "Options:\n"
    "  --cpu               Use CPU (default: CUDA if available)\n"
    "  --no-letterbox      Disable letterbox preprocessing\n"
    "  --classes <path>    Path to classes.txt for YOLO (one label per line)\n"
    "  --size <WxH>        Override network input size (e.g., 640x640 or 518x518)\n"
    "\nExamples:\n"
    "  " << prog << " yolo models/yolov5s.onnx --classes config_files/classes.txt samples/video.mp4\n"
    "  " << prog << " depth models/depth_anything_v2_small.onnx --cpu\n"
  << std::endl;
}

static bool parse_size(const std::string& s, cv::Size& out) {
  // expect "WxH"
  auto x = s.find('x');
  if (x == std::string::npos) return false;
  try {
    int w = std::stoi(s.substr(0, x));
    int h = std::stoi(s.substr(x + 1));
    if (w <= 0 || h <= 0) return false;
    out = cv::Size(w, h);
    return true;
  } catch (...) { return false; }
}

// ---------- App ----------
int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string modeArg   = argv[1];
  const std::string model_path = argv[2];

  bool prefer_cuda   = true;
  bool use_letterbox = true;
  std::string classes_path;     // for YOLO
  cv::Size overrideSize;       // optional
  bool haveOverrideSize = false;
  std::string inputPath;       // optional video path; empty => webcam(0)

  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--cpu") {
      prefer_cuda = false;
    } else if (a == "--no-letterbox") {
      use_letterbox = false;
    } else if (a == "--classes") {
      if (i + 1 >= argc) { std::cerr << "--classes needs a path\n"; return 2; }
      classes_path = argv[++i];
    } else if (a == "--size") {
      if (i + 1 >= argc) { std::cerr << "--size needs WxH\n"; return 2; }
      cv::Size s;
      if (!parse_size(argv[++i], s)) {
        std::cerr << "Invalid --size format. Use WxH (e.g., 640x640)\n";
        return 2;
      }
      overrideSize = s;
      haveOverrideSize = true;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "Unknown option: " << a << "\n";
      print_usage(argv[0]);
      return 2;
    } else {
      inputPath = a; // treat as video source
    }
  }

  // Basic sanity checks
  if (modeArg != "yolo" && modeArg != "depth") {
    std::cerr << "First argument must be 'yolo' or 'depth'\n";
    print_usage(argv[0]);
    return 2;
  }

  // Build the right processor + config
  std::unique_ptr<ImageProcessor> proc;

  if (modeArg == "yolo") {
    YoloConfig configuration;
    configuration.model_path   = model_path;
    configuration.prefer_cuda  = prefer_cuda;
    configuration.use_letterbox= use_letterbox;

    // Reasonable YOLO defaults
    configuration.net_input = haveOverrideSize ? overrideSize : cv::Size(640, 640);
    configuration.scale    = 1.0/255.0;
    configuration.swap_RB   = true;
    configuration.mean     = {0,0,0};
    configuration.std      = {1,1,1};

    // Thresholds (tweak as desired)
    // configuration.confidence_threshold = 0.25f;
    // configuration.nms_threshold  = 0.45f;

    // Load classes if provided; if empty, numeric ids will be displayed
    configuration.classes_path = classes_path;

    auto y = std::make_unique<YoloProcessor>();
    if (!y->load_model(configuration)) {
      std::cerr << "[ERROR] Failed to load YOLO model or get output names\n";
      return 3;
    }
    proc = std::move(y);

  } else { // depth
    DepthConfig configuration;
    configuration.model_path   = model_path;
    configuration.prefer_cuda  = prefer_cuda;
    configuration.use_letterbox= use_letterbox;

    // Depth Anything v2 defaults unless overridden
    configuration.net_input = haveOverrideSize ? overrideSize : cv::Size(518, 518);
    configuration.swap_RB   = true;
    configuration.scale    = 1.0;
    configuration.mean     = {0.485, 0.456, 0.406};
    configuration.std      = {0.229, 0.224, 0.225};

    auto d = std::make_unique<DepthProcessor>();
    if (!d->load_model(configuration)) {
      std::cerr << "[ERROR] Failed to load depth model\n";
      return 3;
    }
    proc = std::move(d);
  }

  // Open input (video or webcam)
  cv::VideoCapture cap;
  if (!inputPath.empty()) {
    cap.open(inputPath);
  } else {
    cap.open(0);
  }
  if (!cap.isOpened()) {
    std::cerr << "[ERROR] Failed to open input: "
              << (inputPath.empty() ? std::string("webcam(0)") : inputPath) << "\n";
    return 4;
  }

  std::cout << "[INFO] Mode: " << modeArg
            << " | Model: " << model_path
            << " | Backend: " << (prefer_cuda ? "CUDA" : "CPU")
            << " | Letterbox: " << (use_letterbox ? "ON" : "OFF")
            << " | Input: " << (inputPath.empty() ? "webcam(0)" : inputPath)
            << std::endl;

  // Main loop
  cv::Mat frame;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      std::cerr << "[WARN] End of stream or failed to read frame.\n";
      break;
    }

    ProcessResult res;
    try {
      res = proc->process(frame);
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] Inference error: " << e.what() << "\n";
      break;
    } catch (...) {
      std::cerr << "[ERROR] Unknown exception during processing.\n";
      break;
    }

    if (res.visualization.empty()) {
      std::cerr << "[WARN] Empty visualizationualization frame.\n";
      continue;
    }

    cv::imshow("Result", res.visualization);
    int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') { // ESC or q
      break;
    }
  }

  return 0;
}