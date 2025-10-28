add_library (image_processor
  # list of cpp source files:
  image_processor.cpp
  depth_processor.cpp
  yolo_processor.cpp
  analyzer.cpp
)

find_package(OpenCV REQUIRED dnn imgproc highgui)

# Indicate what directories should be added to the include file search
# path when using this library.
target_include_directories(image_processor PUBLIC
  # list of directories:
  .
)

target_link_libraries(image_processor PRIVATE
  ${OpenCV_LIBS}
)
