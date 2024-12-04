# Object Re-Identification

## Overview
Feature extraction for object re-identification using TensorRT.

## Export Model
1. Export TorchReID model to ONNX:
```bash
python3 -m venv venv
./venv/bin/pip3 install -r requirement.txt
./venv/bin/python3 torchreid-cli.py -m osnet_x0_25 -e -o model.onnx -s 256 128
```

2. Convert to TensorRT engine:
```shell
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

## Configure
Create `config.json`:
```json
{
  "reid": {
    "engine": {
      "model_path": "path/to/model.engine",
      "batch_size": 1,
      "precision": 16
    },
    "confidence_threshold": 0.5
  }
}
```

## Run
```shell
./build/app/reid/reid -c config.json -i image.jpg
```