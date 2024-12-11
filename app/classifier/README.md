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
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

### Configure
Create `config.json`:
```json
{
  "engine": {
    "model_path": "path/to/model.engine",
    "batch_size": 1,
    "precision": 16
  },
  "confidence_threshold": 0.5,
  "class_names": ["class1", "class2"]
}
```

### Run
```shell
cd build/app/classifier
./classifier -i image.jpg -c config.json -d
```