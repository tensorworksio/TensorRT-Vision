# Object Segmentation

## Overview
Object segmentation engine using TensorRT for optimized inference.

## Supported architectures
### YOLO
- [YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [YOLOv11](https://github.com/ultralytics/ultralytics/tree/main) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

## Configure
In `data/` folder, add your `config.toml`. Class names can be specified as a path to a plain text file (one name per line) or as an inline array.

<details>
    <summary>YOLOv8</summary>

```toml
architecture = "yolo"
name = "yolov8"
confidence_threshold = 0.25
nms_threshold = 0.45
mask_threshold = 0.5
class_names_file = "./data/coco.txt"

[engine]
model_path = "./data/yolov8n-seg.engine"
batch_size = 1
precision = "FP16"
```
</details>
<details open>
    <summary>YOLOv11</summary>

```toml
architecture = "yolo"
name = "yolov11"
confidence_threshold = 0.25
nms_threshold = 0.45
mask_threshold = 0.5
class_names_file = "./data/coco.txt"

[engine]
model_path = "./data/yolo11n-seg.engine"
batch_size = 1
precision = "FP16"
```
</details>

---

## 🖥️ Local

### Build
```shell
# from repo root
meson setup build -Dbuild_apps=segmenter
meson compile -C build
```

### Export model
```shell
python3 -m venv venv
./venv/bin/pip3 install ultralytics onnx onnxsim

mkdir -p app/segmenter/data
./venv/bin/yolo export --model=data/yolo11n-seg.pt --format=onnx --opset=12
trtexec --onnx=data/yolo11n-seg.onnx --saveEngine=data/yolo11n-seg.engine --fp16
```

### Run
```shell
cd build/app/segmenter
./segment -i 0 -o data/webcam.mp4 -c data/config.toml -d
```

---

## 🐳 Docker

### Build
```bash
# from repo root
docker build --target segmenter -t tensorrt-vision:segmenter .
```

### Export model
```shell
python3 -m venv venv
./venv/bin/pip3 install ultralytics onnx onnxsim

mkdir -p data
./venv/bin/yolo export --model=data/yolo11n-seg.pt --format=onnx --opset=12

docker run --gpus all --rm \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/segmenter/data \
    tensorrt-vision:segmenter \
    trtexec --onnx=data/yolo11n-seg.onnx --saveEngine=data/yolo11n-seg.engine --fp16
```

### Run
```bash
xhost +local:docker

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/segmenter/data \
    tensorrt-vision:segmenter \
    ./segment -i data/video.mp4 -c data/config.toml -d
```
