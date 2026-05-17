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
Place your `model.onnx` in `data/`, then convert to a TRT engine:

```bash
docker run --gpus all --rm \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/classifier/data \
    tensorrt-vision:classifier \
    trtexec --onnx=data/model.onnx --saveEngine=data/model.engine --fp16
```

### Run
```bash
xhost +local:docker

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env DISPLAY=$DISPLAY \
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
