# YOLO
TensorRT runtime inference engine from [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api/tree/main) that supports any YOLO version from the following list:
- YOLOv7
- YOLOv8

## Model
```bash
git clone https://github.com/tensorworksio/yolo.cpp.git
cd yolo.cpp && mkdir data
```

Get your __onnx__ model from one of the supported YOLO version and put it on data folder:
- YOLOV7: https://github.com/WongKinYiu/yolov7
- YOLOV8: https://github.com/ultralytics/ultralytics

## Export to TensorRT

```bash
alias trtexec='/usr/src/tensorrt/bin/trtexec'
trtexec --onnx=data/yolov8n.onnx --saveEngine=data/yolov8n_fp16.engine --fp16
```

## Compile
```bash
meson setup build
meson compile -C build
```

## Configure
Configure your YOLO settings in `config.json`

## Run
```bash
./yolo --help
```

### Video
```bash
./yolo -i ../data/video.mp4 -o ../data/out.mp4 --config ../config.json --display
```

### Webcam
```bash
./yolo -i /dev/video0 -c ../config.json --display
```