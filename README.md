# ACME Monocular Perception and Tracking Module
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;&nbsp;&nbsp;&nbsp;
![CICD Workflow status](https://github.com/GraysonGilbert/ACME_perception_module/actions/workflows/run-unit-test-and-upload-codecov.yml/badge.svg) 
&nbsp;&nbsp;&nbsp;&nbsp;
[![codecov](https://codecov.io/gh/GraysonGilbert/ACME_perception_module/graph/badge.svg?token=NOF53QZ586)](https://codecov.io/gh/GraysonGilbert/ACME_perception_module)


## Table of Contents
- [Overview](#overview)
- [Personnel](#personnel)
- [License](#license)
- [AIP Related Documents](#aip-related-documents)
- [Developer / End User Documentation](#developer-documentation)
    - [Camera Calibration](#camera-calibration)
    - [How To Implement Library](#how-to-implement-library)
    - [Dependencies](#dependencies)
    - [Know Bugs / Issues](#known-bugsissues)
    - [How to Build Project](#how-to-build-project)
    - [How to Run Demo](#how-to-run-demo)
    - [How to Run Tests](#how-to-run-tests)
    - [How to Generate Doxygen Docs](#how-to-generate-doxygen-docs)
    - [How to Run Cppcheck](#how-to-run-cppcheck)
    - [How to Build Code Coverage Report](#how-to-build-code-coverage-report)
- [References](#references)
- [Project Deliverables](#project-deliverables)

## Overview

This repository is the midterm assignment for the ENPM700 Software Development course. The following project is a perception module for ACME Robotics. This C++ module will be a human (N>=1) obstacle detector and tracker. It will output location data obtained from analyzing video from a monocular video camera.

#### How does it work?
The module will leverage two separate machine learning models to achieve its main functionality. First, the input image will be processed using the YOLOv5 object detection model. This model will detect and track any person currently in the robot's frame of view. Next, the image will be processed using the Depth Anything monocular depth estimation model. The output from this model will provide the relative depth of each person located in the robot's view. Fusing the results from both models will enable the robot to be able to accurately detect people, perceive where they are located relative to the robot, and track their movement.

#### Module Architecture

![Module Architecture](/UML/revised/ENPM700_midterm_project_phase2.png) 

#### Results
Below is a screenshot of the project demonstration. This is the first frame of the sample video, where both the YOLO detection model and the depth estimation model are applied to the image. The two model results are fused into one output image. The resulting image shows the YOLO human detections and positions, and adds the depth estimation to the detection label.


![Demo Results](/results/demo_results/demo_screenshot.png)


## Personnel

**Grayson Gilbert** - I am a full time Mechanical Engineer currently pursuing a M.Eng in Robotics from the University of Maryland, College Park. I have a broad background involving both hardware and software related projects, and I am interested in working with embedded systems.

**Marcus Hurt** - I am a full time Robotics Engineer currently pursuing a M.Eng in Robotics from the University of Maryland, College Park. I have a Bachelor of Science in Computer Engineering from the University of Michigan, Ann Arbor focusing on embedded systems. I'm interested in deepening my understanding and technical skills in robotics.

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## AIP Related Documents

- **[Proposal](https://docs.google.com/document/d/1lg7BTjVZPU-9VlcfBjtMoFnfqaxiJw49ybgWXe28exQ/edit?usp=sharing)** 
- **[Quad Chart](https://docs.google.com/presentation/d/1ArSuv_W5K5M3QDWWErcLwr7SC7UwhioL13dkYcQ4VEk/edit?usp=sharing)** 
- **[Product Backlog](https://docs.google.com/spreadsheets/d/1AJOp31G0ja_eymtNH2MVXDXD4VEDos6vJx1aDn5YIA0/edit?usp=sharing)**
- **[Sprint 1 Notes](https://docs.google.com/document/d/1XB9LQEMiLpEBPJ9VGN7WRHNV5Q4AIBkNHV68xH3_b5o/edit?usp=sharing)**
- **[Sprint 2 Notes](https://docs.google.com/document/d/1QC9lk04agy9_U9CpSquvnf6A42hWPTi9NtorPjHfCaM/edit?usp=sharing)**

## Developer / End User Documentation

### Camera Calibration
The accuracy of the depth estimation relies on an accurate camera calibration process. To properly calibrate the camera and depth estimation model, the end user must run the model with multiple objetcs of known dimensions at known locations within the camera field of view. Using these known reference locations and object dimensions, a scale factor and offset can be calculated and applied to the model. The scale factor and offset can be modified via the DepthProcessor.hpp file.

### How to Implement Library
The goal of this project is to provide a C++ library that can be implemented in a variety of percpetion applications. The end user should be able to take the library, and create there own applications with its methods for their specific use case. 

### Dependencies

#### OpenCV 4.10
This project depends on using OpenCV. Earlier versions of OpenCV that come standard with Ubuntu 24.04 (i.e. OpenCV 4.6) will not be able to run the ONNX models properly. It is reccomended to build OpenCV 4.10 from source, as that is what was used to develop this project.

#### To build OpenCV 4.10 from source run the following:
```bash
# Install pre-requisites:
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtbbmalloc2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libopenexr-dev libwebp-dev

# Get the sources:
mkdir -p ~/src && cd ~/src
git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv.git
git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv_contrib.git

# Configure, build, and install:
cmake -S opencv -B build-opencv \
  -DOPENCV_EXTRA_MODULES_PATH=~/src/opencv_contrib/modules \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/opencv-4.10 \
  -DBUILD_LIST=core,imgproc,highgui,videoio,dnn \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=ON

cmake --build build-opencv -j"$(nproc)"
sudo cmake --install build-opencv
```
After building from source, you will need to make the packages discoverable. Run the following:
```bash
export PKG_CONFIG_PATH=/opt/opencv-4.10/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/opencv-4.10/lib:$LD_LIBRARY_PATH

pkg-config --modversion opencv4   # should print 4.10.0 (or 4.10.x)
```

#### Git LFS
Another dependency for this project is the git large file storage system. To install git lfs run the following:
```bash
# Update package index
sudo apt update

# Install Git LFS
sudo apt install git-lfs

# Initialize Git LFS in your system
git lfs install

# Verify Installation
git lfs version
```
You should see something like:
```bash
git-lfs/3.5.1 (GitHub; linux amd64; go 1.21)
```

### Known Bugs/Issues

**None...that we know of...yet...**

### How to Build Project
```bash
# Download the code:
  git clone https://github.com/GraysonGilbert/ACME_perception_module.git
  cd ACME_perception_module
  git lfs pull
# Configure the project and generate a native build system:
  # Must re-run this command whenever any CMakeLists.txt file has been changed.
  cmake -S ./ -B build/
# Compile and build the project:
  # rebuild only files that are modified since the last build
  cmake --build build/
  # or rebuild everything from scracth
  cmake --build build/ --clean-first
  # to see verbose output, do:
  cmake --build build/ --verbose
# Clean
  cmake --build build/ --target clean
# Clean and start over:
  rm -rf build/
```

### How to Run Demo
After building the project, from the project root directory, complete the following:
```bash
cd build/ # Navigate to build directory

./app/shell-app # From build directory run the app executable
```

### How to Run Tests

```bash
# Run from project root directory
ctest --test-dir build/
```

### How to Generate Doxygen Docs

```bash
# Run from project root directory

cmake --build build --target docs

# The Doxygen documentation will generate in the /docs subdirectory. 
# To browse the documents in a web browser run:

open build/test_coverage/index.html
```
### How to run cppcheck
```bash
# Run from project root directory
cmake --build build/ --target cppcheck
```
***Note: If running the project in a docker container, you will need to download the /html folder within /docs and open it using your host machine.***

### How to Build Code Coverage Report
```bash
# if you don't have gcovr or lcov installed, do:
  sudo apt-get install gcovr lcov
# Set the build type to Debug and WANT_COVERAGE=ON
  cmake -D WANT_COVERAGE=ON -D CMAKE_BUILD_TYPE=Debug -S ./ -B build/
# Now, do a clean compile, run unit test, and generate the covereage report
  cmake --build build/ --clean-first --target all test_coverage
# open a web browser to browse the test coverage report
  open build/test_coverage/index.html

This generates a index.html page in the build/test_coverage sub-directory that can be viewed locally in a web browser.
```

You can also get code coverage report for the app_coverage target, instead of unit test. Repeat the previous 2 steps but with the app_coverage target:

```bash
# Now, do another clean compile, run pid_controller_example, and generate its covereage report
  cmake --build build/ --clean-first --target all app_coverage
# open a web browser to browse the test coverage report
  open build/app_coverage/index.html

This generates a index.html page in the build/app_coverage sub-directory that can be viewed locally in a web browser.
```

## References
### YOLOv5 Object Detection Model

GitHub - [Link](https://github.com/ultralytics/yolov5)

About YOLOv5 - [Link](https://docs.ultralytics.com/models/yolov5/)


### Depth Anything Monocular Depth Perception Model

Hugging Face Model - [Link](https://huggingface.co/LiheYoung/depth_anything_vitb14)

Research Paper - [Link](https://arxiv.org/abs/2401.10891)

## Project Deliverables
### **Phase 0**
| **Deliverable**      | **Link** |
|:---------------------|:-------- |
| **Proposal**         | **[Link](https://docs.google.com/document/d/1lg7BTjVZPU-9VlcfBjtMoFnfqaxiJw49ybgWXe28exQ/edit?usp=sharing)** |
| **Quad Chart**       | **[Link](https://docs.google.com/presentation/d/1ArSuv_W5K5M3QDWWErcLwr7SC7UwhioL13dkYcQ4VEk/edit?usp=sharing)** |
| **Proposal Video**   | **[Link](https://youtu.be/OkT_tCYltAI?si=i8uGd-1u-vzNCljR)** |
| **Product Backlog**  | **[Link](https://docs.google.com/spreadsheets/d/1AJOp31G0ja_eymtNH2MVXDXD4VEDos6vJx1aDn5YIA0/edit?usp=sharing)** | 
| **UML Class Diagram**| **[Link](https://drive.google.com/file/d/1UlZO1RhkXP9v4lFadjrH4jGhcDqJkhok/view?usp=sharing)**|


### **Phase 1**
| **Deliverable**      | **Link** |
|:---------------------|:-------- |
| **Update Video**   | **[Link](https://youtu.be/76M_7pByoMQ?si=dxlrG3NBnfKesLuL)** |
### **Phase 2**
| **Deliverable**      | **Link** |
|:---------------------|:-------- |
| **Update Video**   |  **[Link](https://youtu.be/L1i_JfZB1Qs?si=KJ57mVypFboSd3uw)**|

