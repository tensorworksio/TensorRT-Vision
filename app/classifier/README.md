# Image Classification

## Overview
Image classification using TensorRT for optimized inference.

## Export Model
1. Export your PyTorch/TensorFlow model to ONNX:
```python
import torch
model = torch.load("model.pt")
torch.onnx.export(model, ...)
```

2. Convert ONNX to TensorRT engine:
```shell
mkdir data
trtexec --onnx=data/model.onnx --saveEngine=data/model.engine --fp16
```

## Configure
In `data` folder, add your `config.json`:
```json
{
  "engine": {
    "model_path": "./data/model.engine",
    "batch_size": 1,
    "precision": 16
  },
  "confidence_threshold": 0.5,
  "class_names": ["class1", "class2"]
}
```

## Compile
```shell
# in root directory
meson setup build -Dbuild_apps=classifier
meson compile -C build
```

## Run

### Display
```shell
# in root directory
cd build/app/classifier
./classify -i image.jpg -c data/config.json -d
```

### JQuery Pipeline
```shell
# in root directory
cd build/app/classifier
./classify -i image.jpg -c data/config.json | jq .data.class_name