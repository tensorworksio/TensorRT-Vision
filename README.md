# TensorRT engine
TensorRT runtime inference engine inspired by [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api/tree/main)

## Export your model
```bash
alias trtexec='/usr/src/tensorrt/bin/trtexec'
trtexec --onnx=data/model.onnx --saveEngine=data/model.engine --fp16
```

## Compile
```bash
meson setup build
meson compile -C build
```

## Configure
Configure your Engine settings in `config.json`


## Example
```bash
./example/classification
```