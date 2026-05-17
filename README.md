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


## TODO
- Prerequisities should not be tied to my system (link to CUDA)
- The slim image cuda does have the necssary trt lib, maybe just use the same image as build and runtime then. Add python deps so that we can install the requirements off all the projects into a venv. We will endup with a tensorrt image that is big but working.

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

All images are built from the single root `Dockerfile` using `--target`.

### Selecting your CUDA + TRT version

The default base image (`nvcr.io/nvidia/tensorrt:25.01-py3`) ships CUDA 12.6 + TRT 10.7 and runs on any GPU whose host driver reports CUDA ≥ 12.6 (`nvidia-smi` shows this).  To use a different version, pass `--build-arg`:

```bash
# Check available TRT tags at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
# Check available CUDA runtime tags at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda

docker build \
  --build-arg TRT_IMAGE=nvcr.io/nvidia/tensorrt:24.09-py3 \
  --build-arg CUDA_IMAGE=nvcr.io/nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 \
  --target detector -t tensorrt-vision:detector .
```

The CUDA_IMAGE must match the **major.minor** CUDA version of TRT_IMAGE.

### Build an app image

```bash
docker build --target <app> -t tensorrt-vision:<app> .
```

Replace `<app>` with: `detector`, `segmenter`, `classifier`, `reid`, or `mot`.

### Export ONNX models

App images do not include Python or `trtexec`. Export your models on the host using each app's local export steps, then mount the `data/` directory when running the container.

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