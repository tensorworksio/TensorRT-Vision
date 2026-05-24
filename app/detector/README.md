# Object Detection

## Overview
Object detection engine using TensorRT for optimized inference.

## Supported architectures
### YOLO
- [YOLOv7](https://github.com/WongKinYiu/yolov7) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [YOLOv11](https://github.com/ultralytics/ultralytics/tree/main) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

## Configure
In `data/` folder, add your `config.toml`. Class names can be specified as a path to a plain text file (one name per line) or as an inline array.

<details>
    <summary>YOLOv7</summary>

```toml
architecture = "yolo"
name = "yolov7"
confidence_threshold = 0.25
nms_threshold = 0.45
class_names_file = "./data/classes.txt"

[engine]
model_path = "./data/yolov7n.engine"
batch_size = 1
precision = "FP16"
```
</details>
<details>
    <summary>YOLOv8</summary>

```toml
architecture = "yolo"
name = "yolov8"
confidence_threshold = 0.25
nms_threshold = 0.45
class_names_file = "./data/classes.txt"

[engine]
model_path = "./data/yolov8n.engine"
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
class_names_file = "./data/coco.txt"

[engine]
model_path = "./data/yolo11n.engine"
batch_size = 1
precision = "FP16"
```
</details>

---

## 🖥️ Local

### Build
```shell
# from repo root
meson setup build -Dbuild_apps=detector
meson compile -C build
```

### Export model
```shell
python3 -m venv venv
./venv/bin/pip3 install ultralytics onnx onnxsim

mkdir -p data
./venv/bin/yolo export --model=data/yolo11n.pt --format=onnx --opset=12
trtexec --onnx=data/yolo11n.onnx --saveEngine=data/yolo11n.engine --fp16
```

### Run
```shell
cd build/app/detector
./detect -i 0 -o data/webcam.mp4 -c data/config.toml -d
```

---

## 🐳 Docker

### Build
```bash
# from repo root
docker build --target detector -t tensorrt-vision:detector .
```

### Export model
`python3.12` and `trtexec` both ship in the image, so the full export — PyTorch → ONNX → TRT engine — runs inside the container. The mounted `data/` volume keeps the generated files on the host.

```bash
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env HOME=/tmp \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/detector/data \
    tensorrt-vision:detector bash -c "\
        python3 -m venv /tmp/venv && \
        /tmp/venv/bin/pip3 install ultralytics onnx onnxsim && \
        /tmp/venv/bin/yolo export --model=data/yolo11n.pt --format=onnx --opset=12 && \
        trtexec --onnx=data/yolo11n.onnx --saveEngine=data/yolo11n.engine --fp16"
```

### Run
The image sets `QT_QPA_PLATFORM=offscreen` so headless runs don't crash; override it with `QT_QPA_PLATFORM=xcb` for a live display window. The webcam (`-i 0`) is passed in with `--device`; `--group-add` gives the unprivileged container user the host's `video` group so it can open the device node.

```bash
xhost +local:

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --device /dev/video0 \
    --group-add $(getent group video | cut -d: -f3) \
    --env DISPLAY=$DISPLAY \
    --env QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/detector/data \
    tensorrt-vision:detector \
    ./detect -i 0 -c data/yolo11.toml -d
```