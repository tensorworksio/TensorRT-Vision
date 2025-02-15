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
- Object Detection
- Multi Object Tracking

## ⚙️ Requirements
1. CUDA 12.6
2. TensorRT 10.7.0
3. Python 3.12.3

Follow installation instructions [here](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

## 🛠️ Build & Install
```bash
# Build all apps
meson setup build -Dbuild_apps=classifier,mot,reid,yolo
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
- [Multi Object Tracking Guide](app/mot/README.md)

## TODO
- Object segmentation app:
    - segmentation folder
    - yolo.cpp impl
    - deteciton: Faster RCNN is a SIMO model -> Detector must not be a SISO that we inherit from, but each model should directly 
      inherit from SISO or SIMOProcessor
    - the same holds for segmentation
    - in segmentation you would have: yolo.cpp but also mask_rcnn.cpp as well (Are the config the same ?)
    - In yolo.cpp: preprocess assumes that a batch contains images of the same sizes which is not necesseraly true !
    
- OCR app