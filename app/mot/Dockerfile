FROM tensorrt-vision:base

# Install dlib from source — cmake generates dlib-1.pc required by mot.cpp
# (Ubuntu's libdlib-dev package does not ship this file)
WORKDIR /tmp
RUN wget -q http://dlib.net/files/dlib-19.6.tar.bz2 \
    && tar xf dlib-19.6.tar.bz2 \
    && cd dlib-19.6 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DDLIB_USE_CUDA=OFF \
    && cmake --build . --config Release -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/dlib-19.6*

WORKDIR /workspace/TensorRT-Vision
RUN meson setup build -Dbuild_apps=mot \
    && ninja -C build -j$(nproc)

# Install Python toolchain for ONNX model export (detector required by mot)
RUN pip install --no-cache-dir ultralytics onnx onnxsim

WORKDIR /workspace/TensorRT-Vision/build/app/mot