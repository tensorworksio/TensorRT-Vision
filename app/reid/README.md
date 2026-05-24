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
`python3.12` and `trtexec` both ship in the image. The export script (`torchreid-cli.py`) is not baked into the image, so mount the repo's `app/reid` directory alongside the `data/` volume, then run the whole export inside the container:

```bash
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env HOME=/tmp \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/reid/data \
    -v $(pwd)/app/reid:/workspace/TensorRT-Vision/app/reid \
    tensorrt-vision:reid bash -c "\
        python3 -m venv /tmp/venv && \
        /tmp/venv/bin/pip3 install -r /workspace/TensorRT-Vision/app/reid/requirements.txt && \
        /tmp/venv/bin/python3 /workspace/TensorRT-Vision/app/reid/torchreid-cli.py -m osnet_x0_25 -e -o data/osnet_x0_25.onnx -s 256 128 && \
        trtexec --onnx=data/osnet_x0_25.onnx --saveEngine=data/osnet_x0_25.engine --fp16"
```

### Run

Display: the image sets `QT_QPA_PLATFORM=offscreen` so headless runs don't crash; override it with `QT_QPA_PLATFORM=xcb` for a live display window.
```bash
xhost +local:

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env DISPLAY=$DISPLAY \
    --env QT_QPA_PLATFORM=xcb \
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
