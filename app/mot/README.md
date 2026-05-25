# Multiple Object Tracking

## Overview
Multiple Object Tracking (MOT) using TensorRT for optimized inference. Supports SORT and BoTSORT trackers with optional ReID feature integration. The trackers are defined in the subproject [mot.cpp](https://github.com/tensorworksio/mot.cpp)

## Supported Trackers
- [SORT](https://github.com/abewley/sort) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [BoTSORT](https://github.com/NirAharon/BoT-SORT) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

## Requirements
1. [Detector](../detector/README.md) or [Segmenter](../segmenter/README.md)
2. [Optional] [ReID](../reid/README.md)

## Configure
Each component has its own TOML config file in the `data/` folder. Class names can be specified as a path to a plain text file (one name per line) or as an inline array.

### Tracker

<details open>
    <summary>SORT</summary>

```toml
tracker = "sort"
max_time_lost = 15
match_thresh = 0.3

[kalman]
time_step = 1
process_noise_scale = 1.0
measurement_noise_scale = 1.0
```
</details>
<details>
    <summary>BoTSORT</summary>

```toml
tracker = "botsort"
max_time_lost = 15
track_high_thresh = 0.5
track_low_thresh = 0.1
new_track_thresh = 0.6
first_match_thresh = 0.3
second_match_thresh = 0.1
unconfirmed_match_thresh = 0.2
proximity_thresh = 0.5
appearance_thresh = 0.9

[kalman]
time_step = 1
process_noise_scale = 1.0
measurement_noise_scale = 1.0
```
</details>

### Detector / Segmenter

See [Detector README](../detector/README.md) and [Segmenter README](../segmenter/README.md) for config examples.

### ReID (optional, recommended with BoTSORT)

```toml
confidence_threshold = 0.8

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
meson setup build -Dbuild_apps=mot
meson compile -C build
```

### Export model
See [Detector README](../detector/README.md) for ONNX export and TRT engine conversion steps.

### Run

<details open>
    <summary>SORT + detector</summary>

```shell
cd build/app/mot
./mot -i 0 -o out.mp4 --tracker data/sort.toml --detector data/yolo11.toml -d
```
</details>
<details>
    <summary>SORT + segmenter</summary>

```shell
cd build/app/mot
./mot -i 0 -o out.mp4 --tracker data/sort.toml --segmenter data/yolo11-seg.toml -d
```
</details>
<details>
    <summary>BoTSORT + detector + ReID</summary>

```shell
cd build/app/mot
./mot -i 0 -o out.mp4 --tracker data/botsort.toml --detector data/yolo11.toml --reid data/osnet.toml -d
```
</details>

---

## 🐳 Docker

### Build
```bash
# from repo root
docker build --target mot -t tensorrt-vision:mot .
```

### Export model
The image bakes in an export virtualenv (`ultralytics`, `onnx`, `trtexec`), so the full export — PyTorch → ONNX → TRT engine — runs inside the container with no installs. The mounted `data/` volume keeps the generated files on the host.

```bash
docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --env HOME=/tmp \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/mot/data \
    tensorrt-vision:mot bash -c "\
        yolo export --model=data/yolo11n.pt --format=onnx --opset=12 && \
        trtexec --onnx=data/yolo11n.onnx --saveEngine=data/yolo11n.engine --fp16"
```

For ReID, follow the [ReID export steps](../reid/README.md#-docker).

### Run
The image sets `QT_QPA_PLATFORM=offscreen` so headless runs don't crash; override it with `QT_QPA_PLATFORM=xcb` for a live display window. The webcam (`-i 0`) is passed in with `--device`; `--group-add` gives the unprivileged container user the host's `video` group so it can open the device node. To track a file instead, drop `--device`/`--group-add` and use `-i data/video.mp4`.

<details open>
    <summary>SORT + detector</summary>

```bash
xhost +local:

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --device /dev/video0 \
    --group-add $(getent group video | cut -d: -f3) \
    --env DISPLAY=$DISPLAY \
    --env QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/mot/data \
    tensorrt-vision:mot \
    ./mot -i 0 -o data/out.mp4 --tracker data/sort.toml --detector data/yolo11.toml -d
```
</details>
<details>
    <summary>BoTSORT + detector + ReID</summary>

```bash
xhost +local:

docker run --gpus all --rm \
    --user $(id -u):$(id -g) \
    --device /dev/video0 \
    --group-add $(getent group video | cut -d: -f3) \
    --env DISPLAY=$DISPLAY \
    --env QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $(pwd)/data:/workspace/TensorRT-Vision/build/app/mot/data \
    tensorrt-vision:mot \
    ./mot -i 0 -o data/out.mp4 --tracker data/botsort.toml --detector data/yolo11.toml --reid data/osnet.toml -d
```
</details>
