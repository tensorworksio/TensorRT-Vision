# Object Re-Identification

## Overview
Feature extraction for object re-identification using TensorRT.

## Export Model
1. Export TorchReID model to ONNX:
```shell
python3 -m venv venv
./venv/bin/pip3 install -r requirement.txt
```

```shell
mkdir data
./venv/bin/python3 torchreid-cli.py -m osnet_x0_25 -e -o data/osnet_x0_25.onnx -s 256 128
```

2. Convert to TensorRT engine:
```shell
trtexec --onnx=data/osnet_x0_25.onnx --saveEngine=data/osnet_x0_25.engine --fp16
```

## Configure
In `data` folder, add your `config.json`:
```json
{
  "reid": {
    "engine": {
      "model_path": "./data/osnet_x0_25.engine",
      "batch_size": 1,
      "precision": 16
    },
    "confidence_threshold": 0.5
  }
}
```

## Compile
```shell
# in root directory
meson setup build -Dbuild_apps=reid
meson compile -C build
```
## Run
```shell
# in root directory
cd build/app/reid
./reid -q image1.jpg -k image2.jpg -c data/config.json -d
```