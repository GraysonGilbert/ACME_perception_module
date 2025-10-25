# ----- Base: Nvidia -----
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 AS base

# ----- Install dependencies -----
RUN set -eux; \
  # Temporarily disable NVIDIA/CUDA repo (keeps Ubuntu 24.04 ubuntu.sources active)
  find /etc/apt/sources.list.d -maxdepth 1 -type f \( -name 'cuda*.list' -o -name 'nvidia*.list' \) \
    -exec sed -i 's/^deb /# deb /' {} +; \
  DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 update; \
  apt-get install -y --no-install-recommends \
    nano \
    git \
    cmake \
    build-essential \
    ccache \
    bear \
    python3 \
    doxygen \
    graphviz \
    libbatik-java \
    default-jre \
    cppcheck \
    clang-format \
    clang-tidy \
    clangd; \
  rm -rf /var/lib/apt/lists/*

# ----- Install OpenCV 4.10 -----
RUN set -eux; \
  apt-get update && \
  apt-get install -y build-essential cmake git pkg-config \
      libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
      libtbbmalloc2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
      libopenexr-dev libwebp-dev && \
  mkdir -p ~/src && cd ~/src && \
  git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv.git && \
  git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv_contrib.git && \
  cmake -S opencv -B build-opencv \
    -DOPENCV_EXTRA_MODULES_PATH=~/src/opencv_contrib/modules \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/opencv-4.10 \
    -DBUILD_LIST=core,imgproc,highgui,videoio,dnn \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON && \
  cmake --build build-opencv -j"$(nproc)" && \
  cmake --install build-opencv && \
  rm -rf /var/lib/apt/lists/*

ENV PKG_CONFIG_PATH=/opt/opencv-4.10/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/opencv-4.10/lib:$LD_LIBRARY_PATH
