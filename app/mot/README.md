# Multiple Object Tracking

## Overview
Multiple Object Tracking (MOT) using TensorRT for optimized inference. Supports SORT and BoTSORT trackers with optional ReID feature integration.

## Supported Trackers
- [SORT](https://github.com/abewley/sort) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)
- [BoTSORT](https://github.com/NirAharon/BoT-SORT) ![Support](https://img.shields.io/badge/support-yes-brightgreen.svg)

## Requirements
1. Detector model: [YOLO](../yolo/README.md)
2. (Optional) ReId model: [ReId](../reid/README.md)

## Configure
In `data` folder, add your `config.json`:

<details>
    <summary>SORT</summary>

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
"reid": {
    "engine": {
        "model_path": "../reid/data/osnet_x0_25.engine",
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
        "model_path": "../yolo/data/yolo11n.engine",
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
                "model_path": "../reid/data/osnet_x0_25.engine",
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
                "model_path": "../yolo/data/yolo11n.engine",
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
./mot -i 0 -o data/webcam.mp4 -c data/config.json -d
```