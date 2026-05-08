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
In `data` folder, add your `config.json`. Class names can be specified as a path to a plain text file (one name per line) or as an inline array.

```json
{
  "engine": {
    "model_path": "./data/model.engine",
    "batch_size": 1,
    "precision": 16
  },
  "confidence_threshold": 0.5,
  "class_names_file": "./data/classes.txt"
}
```

Config files support `//` and `/* */` comments.

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
```
