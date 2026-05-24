# Image Classification

## Overview
Image classification using TensorRT for optimized inference.

## Configure
In `data/` folder, add your `config.toml`. Class names can be specified as a path to a plain text file (one name per line) or as an inline array.

```toml
confidence_threshold = 0.5
class_names_file = "./data/classes.txt"

[engine]
model_path = "./data/model.engine"
batch_size = 1
precision = "FP16"
```

---

## 🖥️ Local

### Build
```shell
# from repo root
meson setup build -Dbuild_apps=classifier
meson compile -C build
```

### Export model
Export your PyTorch/TensorFlow model to ONNX, then convert to a TensorRT engine:

```python
import torch
model = torch.load("model.pt")
torch.onnx.export(model, ...)
```

```shell
mkdir -p app/classifier/data
trtexec --onnx=data/model.onnx --saveEngine=data/model.engine --fp16
```

### Run

Display:
```shell
cd build/app/classifier
./classify -i image.jpg -c data/config.toml -d
```

JSON pipeline:
```shell
cd build/app/classifier
./classify -i image.jpg -c data/config.toml | jq .data.class_name
```

---

## 🐳 Docker

### Build
```bash
# from repo root
docker build --target classifier -t tensorrt-vision:classifier .
```

### Export model
`python3.12` and `trtexec` both ship in the image. Place your `model.onnx` in `data/`, then convert it to a TRT engine inside the container:

```bash
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/classifier/data \
    tensorrt-vision:classifier \
    trtexec --onnx=data/model.onnx --saveEngine=data/model.engine --fp16
```

To run the PyTorch/TensorFlow → ONNX export in-container too, create a venv first and install your framework, e.g.:

```bash
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env HOME=/tmp \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/classifier/data \
    tensorrt-vision:classifier bash -c "\
        python3 -m venv /tmp/venv && \
        /tmp/venv/bin/pip3 install torch onnx && \
        /tmp/venv/bin/python3 data/export.py && \
        trtexec --onnx=data/model.onnx --saveEngine=data/model.engine --fp16"
```

### Run
The image sets `QT_QPA_PLATFORM=offscreen` so headless runs don't crash; override it with `QT_QPA_PLATFORM=xcb` for a live display window.

```bash
xhost +local:

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env DISPLAY=$DISPLAY \
    --env QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/classifier/data \
    tensorrt-vision:classifier \
    ./classify -i data/image.jpg -c data/config.toml -d
```

JSON pipeline:
```bash
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/classifier/data \
    tensorrt-vision:classifier \
    ./classify -i data/image.jpg -c data/config.toml | jq .data.class_name
```
