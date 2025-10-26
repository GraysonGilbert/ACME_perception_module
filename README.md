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
- [Architecture](#architecture)
- [Developer Documentation](#developer-documentation)
    - [Dependencies](#dependencies)
    - [How to Build Project](#how-to-build-project)
    - [How to Build Code Coverage Report](#how-to-build-code-coverage-report)
    - [How to Run Demo](#how-to-run-demo)
    - [How to Run Tests](#how-to-run-tests)
    - [How to Generate Doxygen Docs](#how-to-generate-doxygen-docs)
- [Project Deliverables](#project-deliverables)

## Overview

This repository is the midterm assignment for the ENPM700 Software Development course. The following project is a perception module for ACME Robotics. This C++ module will be a human (N>=1) obstacle detector and tracker. It will output location data obtained from analyzing video from a monocular video camera.

#### How does it work?
The module will leverage two separate machine learning models to achieve its main functionality. First, the input image will be processed using the YOLOv5 object detection model. This model will detect and track any person currently in the robot's frame of view. Next, the image will be processed using the Depth Anything monocular depth estimation model. The output from this model will provide the relative depth of each person located in the robot's view. Fusing the results from both models will enable the robot to be able to accurately detect people, perceive where they are located relative to the robot, and track their movement.

#### Module Architecture

![Module Architecture](/UML/initial/ENPM700_midterm_project_class_UML.png) 

#### Results

[Include Results from Demonstration]



 

## Personnel

**Grayson Gilbert** - I am a full time Mechanical Engineer currently pursuing a M.Eng in Robotics from the University of Maryland, College Park. I have a broad background involving both hardware and software related projects, and I am interested in working with embedded systems.

**Marcus Hurt** - [Insert personal statement here.]

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## AIP Related Documents

- **[Proposal](https://docs.google.com/document/d/1lg7BTjVZPU-9VlcfBjtMoFnfqaxiJw49ybgWXe28exQ/edit?usp=sharing)** 
- **[Quad Chart](https://docs.google.com/presentation/d/1ArSuv_W5K5M3QDWWErcLwr7SC7UwhioL13dkYcQ4VEk/edit?usp=sharing)** 
- **[Product Backlog](https://docs.google.com/spreadsheets/d/1AJOp31G0ja_eymtNH2MVXDXD4VEDos6vJx1aDn5YIA0/edit?usp=sharing)**

## Developer Documentation

### Dependencies

### Known Bugs/Issues

### How to Build Project
```bash
# Download the code:
  git clone https://github.com/GraysonGilbert/ACME_perception_module.git
  cd ACME_perception_module
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
# Run program:
  ./build/app/shell-app
# Run tests:
  cd build/; ctest; cd -
  # or if you have newer cmake
  ctest --test-dir build/
# Build docs:
  cmake --build build/ --target docs
  # open a web browser to browse the doc
  open docs/html/index.html
# Clean
  cmake --build build/ --target clean
# Clean and start over:
  rm -rf build/
```
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

You can also get code coverage report for the [ **Update Target Name Here** ] target, instead of unit test. Repeat the previous 2 steps but with the app_coverage target:

```bash
# Now, do another clean compile, run pid_controller_example, and generate its covereage report
  cmake --build build/ --clean-first --target all app_coverage
# open a web browser to browse the test coverage report
  open build/app_coverage/index.html

This generates a index.html page in the build/app_coverage sub-directory that can be viewed locally in a web browser.
```

### How to Run Demo

### How to Run Tests

### How to Generate Doxygen Docs


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


### **Phase 2**