FROM tensorrt-vision:base

RUN meson setup build -Dbuild_apps=classifier \
    && ninja -C build -j$(nproc)

WORKDIR /workspace/TensorRT-Vision/build/app/classifier