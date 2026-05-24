<div align="center">

# TensorRT-Vision
### A TensorRT Toolbox for Optimized Vision Model Inference

[![python](https://img.shields.io/badge/python-3.12-green)](https://www.python.org/downloads/release/python-3123/)
[![cuda](https://img.shields.io/badge/CUDA-driver%20dependent-blue)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TensorRT-driver%20dependent-blue)](https://developer.nvidia.com/tensorrt)

</div>

## 📋 Overview
TensorRT-Vision provides optimized inference for computer vision models using NVIDIA TensorRT. It supports:

- Object Detection
- Object Segmentation
- Object Classification
- Object Re-Identification
- Multi Object Tracking


## ⚙️ Prerequisites

There is no single fixed CUDA/TensorRT requirement — the right stack depends on your GPU driver. Run `nvidia-smi` and read the **CUDA Version** in the top-right: that is the highest CUDA your driver supports. Pick a CUDA + TensorRT combination at or below it.

### 🖥️ Local
1. Python 3.12
2. A CUDA Toolkit supported by your GPU driver
3. The matching TensorRT release for that CUDA version

Follow installation instructions [here](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

### 🐳 Docker
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so containers can access the GPU. The CUDA + TensorRT stack is then fully determined by the NGC base image tag chosen at build time — see [Selecting your CUDA + TRT version](#selecting-your-cuda--trt-version). Nothing else to install on the host.

## 🖥️ Local Build

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

The default base image (`nvcr.io/nvidia/tensorrt:25.01-py3`) ships CUDA 12.6 + TRT 10.7 and runs on any GPU whose host driver reports CUDA ≥ 12.6 (`nvidia-smi` shows this).  The same image backs both the build and runtime stages, so a single `--build-arg` switches versions:

```bash
# Check available TRT tags at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
docker build \
  --build-arg TRT_IMAGE=nvcr.io/nvidia/tensorrt:24.09-py3 \
  --target detector -t tensorrt-vision:detector .
```

### Build an app image

```bash
docker build --target <app> -t tensorrt-vision:<app> .
```

Replace `<app>` with: `detector`, `segmenter`, `classifier`, `reid`, or `mot`.

### Export ONNX models

App images include `python3.12` and `trtexec`, so models can be exported to ONNX and converted to TRT engines entirely inside the container — no host Python needed. Mount the `data/` directory so the generated engine persists on the host. See each app's README for the exact command.

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