# YOLO Object Detection

## Overview
YOLO object detection using TensorRT for optimized inference.

## Supported Versions
- [YOLOv7](https://github.com/WongKinYiu/yolov7) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [YOLOv11](https://github.com/ultralytics/ultralytics/tree/main) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

## Export Model
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
```json
{
  "detector": {
    "name": "yolov11",
    "confidence_threshold": 0.25,
    "nms_threshold": 0.45,
    "engine": {
      "model_path": "./data/yolo11n.engine",
      "batch_size": 1,
      "precision": 16
    },
    "class_names": [
      "person",
      "bicycle",
      "car",
      "motorbike",
      "aeroplane",
      "bus",
      "train",
      "truck",
      "boat",
      "traffic light",
      "fire hydrant",
      "stop sign",
      "parking meter",
      "bench",
      "bird",
      "cat",
      "dog",
      "horse",
      "sheep",
      "cow",
      "elephant",
      "bear",
      "zebra",
      "giraffe",
      "backpack",
      "umbrella",
      "handbag",
      "tie",
      "suitcase",
      "frisbee",
      "skis",
      "snowboard",
      "sports ball",
      "kite",
      "baseball bat",
      "baseball glove",
      "skateboard",
      "surfboard",
      "tennis racket",
      "bottle",
      "wine glass",
      "cup",
      "fork",
      "knife",
      "spoon",
      "bowl",
      "banana",
      "apple",
      "sandwich",
      "orange",
      "broccoli",
      "carrot",
      "hot dog",
      "pizza",
      "donut",
      "cake",
      "chair",
      "sofa",
      "pottedplant",
      "bed",
      "diningtable",
      "toilet",
      "tvmonitor",
      "laptop",
      "mouse",
      "remote",
      "keyboard",
      "cell phone",
      "microwave",
      "oven",
      "toaster",
      "sink",
      "refrigerator",
      "book",
      "clock",
      "vase",
      "scissors",
      "teddy bear",
      "hair drier",
      "toothbrush"
    ]
  }
}
```

## Compile
```shell
# in root directory
meson setup build -Dbuild_apps=yolo
meson compile -C build
```

## Run
```shell
# in root directory
cd build/app/yolo
./yolo -i 0 -o data/webcam.mp4 -c data/config.json -d
```