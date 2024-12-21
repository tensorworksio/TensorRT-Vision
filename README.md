<div align="center">

# TensorRT-Vision
### A TensorRT Toolbox for Optimized Vision Model Inference

[![python](https://img.shields.io/badge/python-3.12.3-green)](https://www.python.org/downloads/release/python-3123/)
[![cuda](https://img.shields.io/badge/cuda-12.6-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.5.0-green)](https://developer.nvidia.com/tensorrt)

</div>

## 📋 Overview
TensorRT-Vision provides optimized inference for computer vision models using NVIDIA TensorRT. It supports:

- Object Classification
- Object Re-Identification  
- Object Detection (YOLO)

## ⚙️ Requirements
1. CUDA 12.6
2. TensorRT 10.7.0
3. Python 3.12.3

Follow installation instructions [here](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

## 🛠️ Build & Install
```bash
# Build all apps
meson setup build -Dbuild_apps=yolo,reid,classifier
meson compile -C build

# Or build a specific app
meson setup build -Dbuild_apps=yolo
meson compile -C build

# Make sure trtexec is installed for model export
alias trtexec='/usr/src/tensorrt/bin/trtexec'
```

## 🚀 Quick Start
Each app has its own README with detailed instructions:

- [Classification Guide](app/classifier/README.md)
- [Re-Identification Guide](app/reid/README.md)
- [YOLO Detection Guide](app/yolo/README.md)