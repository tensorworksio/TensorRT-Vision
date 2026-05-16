FROM tensorrt-vision:base

RUN meson setup build -Dbuild_apps=segmenter \
    && ninja -C build -j$(nproc)

# Install Python toolchain for ONNX model export
RUN pip install --no-cache-dir ultralytics onnx onnxsim

WORKDIR /workspace/TensorRT-Vision/build/app/segmenter