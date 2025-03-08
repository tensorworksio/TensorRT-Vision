# Multiple Object Tracking

## Overview
Multiple Object Tracking (MOT) using TensorRT for optimized inference. Supports SORT and BoTSORT trackers with optional ReID feature integration. The trackers are defined in the subproject [mot.cpp](https://github.com/tensorworksio/mot.cpp)

## Supported Trackers
- [SORT](https://github.com/abewley/sort) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [BoTSORT](https://github.com/NirAharon/BoT-SORT) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

## Requirements
1. [Detector](../detector/README.md) or [Segmenter](../segmenter/README.md) 
2. [Optional] [ReId](../reid/README.md)

## Configure
In `data` folder, add your `config.json`:

<details open>
    <summary>SORT + detector</summary>

```json
{
"tracker": {
    "name": "sort",
    "kalman": {
        "time_step": 1,
        "process_noise_scale": 1.0,
        "measurement_noise_scale": 1.0
    },
    "max_time_lost": 15,
    "match_thresh": 0.3
},
"detector": {
    "architecture": "yolo",
    "name": "yolov11",
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
<details>
    <summary>SORT + segmenter</summary>

```json
{
"tracker": {
    "name": "sort",
    "kalman": {
        "time_step": 1,
        "process_noise_scale": 1.0,
        "measurement_noise_scale": 1.0
    },
    "max_time_lost": 15,
    "match_thresh": 0.3
},
"segmenter": {
    "architecture": "yolo",
    "name": "yolov11",
    "confidence_threshold": 0.25,
    "nms_threshold": 0.45,
    "mask_threshold": 0.5,
    "engine": {
        "model_path": "./data/yolo11n-seg.engine",
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
    <summary>BoTSORT</summary>

```json
{
"tracker": {
    "name": "botsort",
    "kalman": {
            "time_step": 1,
            "process_noise_scale": 1.0,
            "measurement_noise_scale": 1.0
    },
    "max_time_lost": 15,
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "first_match_thresh": 0.3,
    "second_match_thresh": 0.1,
    "unconfirmed_match_thresh": 0.2,
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.9
},
"reid": {
    "engine": {
            "model_path": "./data/osnet_x0_25.engine",
            "batch_size": 1,
            "precision": 16
    },
    "confidence_threshold": 0.8
},
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
        // fill in the class names
    ]
}
}
```
</details>

## Compile
```shell
# in root directory
meson setup build -Dbuild_apps=mot
meson compile -C build
```

## Run
```shell
# in root directory
cd build/app/mot
./mot -i 0 -o out.mp4 -c data/config.json -d
```