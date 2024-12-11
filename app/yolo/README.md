# YOLO Object Detection

## Overview
YOLO object detection using TensorRT for optimized inference.

## Supported Versions
- [Yolov7](https://github.com/WongKinYiu/yolov7) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [Yolov8](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [Yolov11](https://github.com/ultralytics/ultralytics/tree/main) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

## Export Model
1. Export YOLO model to ONNX:
```shell
python3 -m venv venv
./venv/bin/pip3 install ultralytics onnx onnxsim
./venv/bin/yolo export --model=yolov8n.pt --format=onnx --opset=12
```

2. Convert to TensorRT engine:
```shell
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

## Configure
Create `config.json`:
```json
{
  "yolo": {
    "engine": {
      "model_path": "path/to/yolo.engine",
      "batch_size": 1,
      "precision": 16
    },
    "version": 8,
    "probability_threshold": 0.25,
    "nms_threshold": 0.45,
    "class_names": ["class1", "class2"]
  }
}
```

## Run
```shell
./build/app/yolo/yolo -c config.json -i image.jpg
```