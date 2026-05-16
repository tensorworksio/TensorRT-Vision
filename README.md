<div align="center">

# TensorRT-Vision
### A TensorRT Toolbox for Optimized Vision Model Inference

[![python](https://img.shields.io/badge/python-3.12.3-green)](https://www.python.org/downloads/release/python-3123/)
[![cuda](https://img.shields.io/badge/cuda-12.6-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.5.0-green)](https://developer.nvidia.com/tensorrt)

</div>

## 📋 Overview
TensorRT-Vision provides optimized inference for computer vision models using NVIDIA TensorRT. It supports:

- Object Detection
- Object Segmentation
- Object Classification
- Object Re-Identification
- Multi Object Tracking

## ⚙️ Prerequisites

### Local
1. CUDA 12.6
2. TensorRT 10.7.0
3. Python 3.12.3

Follow installation instructions [here](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

### Docker
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so Docker containers can access the GPU.

## 🛠️ Local Build

```bash
# Build all apps (default)
meson setup build
meson compile -C build

# Or build specific apps
meson setup build -Dbuild_apps=detector,mot
meson compile -C build

# Make sure trtexec is available for model conversion
alias trtexec='/usr/src/tensorrt/bin/trtexec'
```

## 🐳 Docker Build

The project uses a two-level Docker hierarchy: a shared **base image** with all common dependencies, and one **app image** per application that inherits from it.

**Step 1 — build the base image (once, shared by all apps):**
```bash
docker build -t tensorrt-vision:base .
```

**Step 2 — build the app image:**
```bash
docker build -t tensorrt-vision:<app> -f app/<app>/Dockerfile .
```

Replace `<app>` with: `detector`, `segmenter`, `classifier`, `reid`, or `mot`.

## 🚀 Quick Start
Each app has its own README with detailed local and Docker instructions:

- [Object Detection](app/detector/README.md)
- [Object Segmentation](app/segmenter/README.md)
- [Multi Object Tracking](app/mot/README.md)
- [Object Classification](app/classifier/README.md)
- [Object Re-Identification](app/reid/README.md)

## 🙏 Credits

This project builds upon foundations from:
- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) - A C++ TensorRT wrapper