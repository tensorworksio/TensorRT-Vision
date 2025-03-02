# Object Detection

## Overview
Object detection engine using TensorRT for optimized inference.

## Supported architectures
### YOLO
- [YOLOv7](https://github.com/WongKinYiu/yolov7) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [YOLOv11](https://github.com/ultralytics/ultralytics/tree/main) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

1. Export YOLO model to ONNX:
```shell
python3 -m venv venv
./venv/bin/pip3 install ultralytics onnx onnxsim
```

```shell
mkdir data
./venv/bin/yolo export --model=data/yolo11n.pt --format=onnx --opset=12
```

2. Convert to TensorRT engine:
```shell
trtexec --onnx=data/yolo11n.onnx --saveEngine=data/yolo11n.engine --fp16
```

## Configure
In `data` folder, add your `config.json`:
<details>
    <summary>YOLOv7</summary>

```json
{
  "detector": {
    "architecture": "yolo",
    "version": "yolov7",
    "confidence_threshold": 0.25,
    "nms_threshold": 0.45,
    "engine": {
      "model_path": "./data/yolov7n.engine",
      "batch_size": 1,
      "precision": 16
    },
    "class_names": [
      // fill in the class names
    ]
  }
}
```
</details>
<details>
    <summary>YOLOv8</summary>

```json
{
  "detector": {
    "architecture": "yolo",
    "version": "yolov8",
    "confidence_threshold": 0.25,
    "nms_threshold": 0.45,
    "engine": {
      "model_path": "./data/yolov8n.engine",
      "batch_size": 1,
      "precision": 16
    },
    "class_names": [
      // fill in the class names
    ]
  }
}
```
</details>
<details open>
    <summary>YOLOv11</summary>

```json
{
  "detector": {
    "architecture": "yolo",
    "version": "yolov11",
    "confidence_threshold": 0.25,
    "nms_threshold": 0.45,
    "engine": {
      "model_path": "./data/yolo11n.engine",
      "batch_size": 1,
      "precision": 16
    },
    "class_names": [
      // fill in the class names
    ]
  }
}
```
</details>

## Compile
```shell
# in root directory
meson setup build -Dbuild_apps=detector
meson compile -C build
```

## Run
```shell
# in root directory
cd build/app/detector
./detect -i 0 -o data/webcam.mp4 -c data/config.json -d
```