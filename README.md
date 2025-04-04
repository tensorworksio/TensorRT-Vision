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

## ⚙️ Requirements
1. CUDA 12.6
2. TensorRT 10.7.0
3. Python 3.12.3

Follow installation instructions [here](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

## 🛠️ Build & Install
```bash
# Build all apps (default)
meson setup build
meson compile -C build

# Or explicitly specify apps to build
meson setup build -Dbuild_apps=detector,mot
meson compile -C build

# Make sure trtexec is installed for model export
alias trtexec='/usr/src/tensorrt/bin/trtexec'
```

## 🚀 Quick Start
Each app has its own README with detailed instructions:

- [Object Detection Guide](app/detector/README.md)
- [Object Segmentation Guide](app/segmenter/README.md)
- [Multi Object Tracking Guide](app/mot/README.md)
- [Object Classification Guide](app/classifier/README.md)
- [Object Re-Identification Guide](app/reid/README.md)

## 🙏 Credits

This project builds upon foundations from:
- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) - A C++ TensorRT wrapper