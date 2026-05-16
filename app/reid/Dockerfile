FROM tensorrt-vision:base

RUN meson setup build -Dbuild_apps=reid \
    && ninja -C build -j$(nproc)

# Install Python toolchain for TorchReID ONNX model export
RUN pip install --no-cache-dir -r /workspace/TensorRT-Vision/app/reid/requirements.txt

WORKDIR /workspace/TensorRT-Vision/build/app/reid