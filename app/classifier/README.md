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
In `data` folder, add your `config.toml`. Class names can be specified as a path to a plain text file (one name per line) or as an inline array.

```toml
confidence_threshold = 0.5
class_names_file = "./data/classes.txt"

[engine]
model_path = "./data/model.engine"
batch_size = 1
precision = "FP16"
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
./classify -i image.jpg -c data/config.toml -d
```

### JQuery Pipeline
```shell
# in root directory
cd build/app/classifier
./classify -i image.jpg -c data/config.toml | jq .data.class_name
```
