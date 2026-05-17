# Object Re-Identification

## Overview
Feature extraction for object re-identification using TensorRT.

## Configure
In `data/` folder, add your `config.toml`:

```toml
confidence_threshold = 0.5

[engine]
model_path = "./data/osnet_x0_25.engine"
batch_size = 1
precision = "FP16"
```

---

## 🖥️ Local

### Build
```shell
# from repo root
meson setup build -Dbuild_apps=reid
meson compile -C build
```

### Export model
```shell
python3 -m venv venv
./venv/bin/pip3 install -r app/reid/requirements.txt

mkdir -p app/reid/data
./venv/bin/python3 app/reid/torchreid-cli.py -m osnet_x0_25 -e -o data/osnet_x0_25.onnx -s 256 128
trtexec --onnx=data/osnet_x0_25.onnx --saveEngine=data/osnet_x0_25.engine --fp16
```

### Run

Display:
```shell
cd build/app/reid
./reid -q image1.jpg -k image2.jpg -c data/config.toml -d
```

JSON pipeline:
```shell
cd build/app/reid
./reid -q image1.jpg -k image2.jpg -c data/config.toml | jq .data.match
```

---

## 🐳 Docker

### Build
```bash
# from repo root
docker build --target reid -t tensorrt-vision:reid .
```

### Export model
```shell
python3 -m venv venv
./venv/bin/pip3 install -r app/reid/requirements.txt

mkdir -p data
./venv/bin/python3 app/reid/torchreid-cli.py -m osnet_x0_25 -e -o data/osnet_x0_25.onnx -s 256 128

docker run --gpus all --rm \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/reid/data \
    tensorrt-vision:reid \
    trtexec --onnx=data/osnet_x0_25.onnx --saveEngine=data/osnet_x0_25.engine --fp16
```

### Run

Display:
```bash
xhost +local:docker

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/reid/data \
    tensorrt-vision:reid \
    ./reid -q data/image1.jpg -k data/image2.jpg -c data/config.toml -d
```

JSON pipeline:
```bash
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/reid/data \
    tensorrt-vision:reid \
    ./reid -q data/image1.jpg -k data/image2.jpg -c data/config.toml | jq .data.match
```
