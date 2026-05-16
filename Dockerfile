FROM nvcr.io/nvidia/tensorrt:25.01-py3

# <print> header (used by the apps) requires GCC 14+.
# Ubuntu 24.04 ships GCC 13; GCC 14 is available in the universe repository.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-14 \
    g++-14 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100 \
    && rm -rf /var/lib/apt/lists/*

# Build tools + system libraries shared by all apps
#
# pkg-config    : meson dependency resolver (opencv4, spdlog, gtest)
# ninja-build   : meson default backend
# cmake         : required by some cmake-based subproject deps
# libopencv-dev : opencv4 (core, highgui, imgproc, imgcodecs, video, videoio)
# libspdlog-dev : structured logging
# libgtest-dev  : GTest / GTest_main (vision-core unit tests)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    pkg-config \
    ninja-build \
    libopencv-dev \
    libspdlog-dev \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04's apt meson is 1.3; pip gets the latest
RUN pip install --no-cache-dir "meson>=1.3"

# /usr/local/cuda/lib64 is in ldconfig's runtime cache but not in the compile-time
# linker search path. LIBRARY_PATH tells GCC/ld where to find libs at link time.
ENV LIBRARY_PATH="/usr/local/cuda/lib64"

WORKDIR /workspace
COPY . TensorRT-Vision/
WORKDIR /workspace/TensorRT-Vision
