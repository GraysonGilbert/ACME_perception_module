# Any C++ source files needed to build this target (cpp-test).
add_executable(cpp-test
  # list of source cpp files:
  main.cpp
  test_analyzer.cpp
  test_depth_processor.cpp
  test_letterbox.cpp
  test_yolo_processor.cpp
)


# Any include directories needed to build this target.
# Note: we do not need to specify the include directories for the
# dependent libraries, they are automatically included.
target_include_directories(cpp-test PUBLIC
  # list of include directories:
  ${CMAKE_SOURCE_DIR}/include
)

# Any dependent libraires needed to build this target.
find_package(OpenCV REQUIRED dnn imgproc highgui)

target_link_libraries(cpp-test PUBLIC
  # list of libraries:
  gtest
  image_processor
  ${OpenCV_LIBS}
)

# Enable CMake's test runner to discover the tests included in the
# binary, using the GoogleTest CMake module.
gtest_discover_tests(cpp-test)