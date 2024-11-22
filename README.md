<div align="center">

TensorRT-Vision
===========================
<h4> A TensorRT Toolbox for Optimized Vision Model Inference</h4>

[![python](https://img.shields.io/badge/python-3.12.3-green)](https://www.python.org/downloads/release/python-3123/)
[![cuda](https://img.shields.io/badge/cuda-12.6-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.5.0-green)](https://developer.nvidia.com/tensorrt)
---
<div align="left">

## Requirements
Install CUDA and TensorRT by following the instructions [here](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

## Export your model
```bash
alias trtexec='/usr/src/tensorrt/bin/trtexec'
trtexec --onnx=data/model.onnx --saveEngine=data/model.engine --fp16
```

## Compile
```bash
meson setup build
meson compile -C build
```

## Configure
Configure your Engine settings in `config.json`


## Apps
```bash
- [x] Object classification
- [x] Object re-identification
- [ ] Object detection
- [ ] YOLO
```