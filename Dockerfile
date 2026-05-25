# ─── Configuration ────────────────────────────────────────────────────────────
#
# TRT_IMAGE: NGC TensorRT image, used for BOTH the build and runtime stages.
# It ships TensorRT, CUDA, trtexec and python3.12 — everything the apps need to
# build their own engines and run inference — so there is no separate slim base.
# Tag format: YY.MM-py3  →  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
#
# Common tags and what they ship:
#   24.09-py3  →  CUDA 12.6  +  TRT 10.5  +  Ubuntu 22.04
#   25.01-py3  →  CUDA 12.6  +  TRT 10.7  +  Ubuntu 24.04
#   25.04-py3  →  CUDA 12.9  +  TRT 10.9  +  Ubuntu 24.04
#
# To select the right tag: check your host driver version with `nvidia-smi`,
# which shows the maximum CUDA version the driver supports.  Any TRT tag whose
# CUDA version is ≤ that value will work on your GPU.
#
ARG TRT_IMAGE=nvcr.io/nvidia/tensorrt:25.01-py3

# ─── Stage 1: builder ─────────────────────────────────────────────────────────
FROM ${TRT_IMAGE} AS builder

# Ubuntu 24.04 defaults to GCC 13; the <print> C++23 header requires GCC 14.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-14 g++-14 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake meson pkg-config ninja-build \
    libopencv-dev libopencv-highgui-dev libspdlog-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# /usr/local/cuda/lib64 is in ldconfig's runtime cache but not the compile-time
# linker search path.
ENV LIBRARY_PATH="/usr/local/cuda/lib64"

WORKDIR /workspace
COPY . TensorRT-Vision/
WORKDIR /workspace/TensorRT-Vision

# ─── Stage 2: build-classifier ────────────────────────────────────────────────
FROM builder AS build-classifier
RUN meson setup build -Dbuild_apps=classifier && ninja -C build -j$(nproc)

# ─── Stage 3: build-segmenter ─────────────────────────────────────────────────
FROM builder AS build-segmenter
RUN meson setup build -Dbuild_apps=segmenter && ninja -C build -j$(nproc)

# ─── Stage 4: build-reid ──────────────────────────────────────────────────────
FROM builder AS build-reid
RUN meson setup build -Dbuild_apps=reid && ninja -C build -j$(nproc)

# ─── Stage 5: build-detector ──────────────────────────────────────────────────
FROM builder AS build-detector
RUN meson setup build -Dbuild_apps=detector && ninja -C build -j$(nproc)

# ─── Stage 6: build-mot ───────────────────────────────────────────────────────
FROM builder AS build-mot
RUN meson setup build -Dbuild_apps=mot && ninja -C build -j$(nproc)

# ─── Stage 7: runtime-base ────────────────────────────────────────────────────
# Same TRT image as the builder (TRT libs, CUDA, trtexec, python3.12 all present)
# but WITHOUT the compiler/build tooling.  OpenCV and spdlog come from apt, which
# resolves the full transitive closure (ffmpeg codecs, GDAL, GDCM, Qt5, etc.) —
# no manual .so copying.
FROM ${TRT_IMAGE} AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev libopencv-highgui-dev libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

# The NGC image ships python3.12 but not the venv module (ensurepip); add it so
# we can build the model-export virtualenv below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Bake a model-export virtualenv into the image so users don't reinstall the
# heavy PyTorch/ultralytics stack on every export. Torch is CPU-only: ONNX
# export runs on CPU and the image already ships CUDA/TRT for the trtexec step.
ENV EXPORT_VENV=/opt/export-venv
RUN python3 -m venv ${EXPORT_VENV} \
    && ${EXPORT_VENV}/bin/pip install --no-cache-dir --upgrade pip \
    && ${EXPORT_VENV}/bin/pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && ${EXPORT_VENV}/bin/pip install --no-cache-dir ultralytics onnx onnxsim onnxruntime

# Put the export venv (yolo, python3) and trtexec on PATH for engine builds.
ENV PATH="${EXPORT_VENV}/bin:/usr/src/tensorrt/bin:${PATH}"
ENV QT_QPA_PLATFORM=offscreen
WORKDIR /workspace/TensorRT-Vision

# ─── Stage 8: classifier ──────────────────────────────────────────────────────
FROM runtime-base AS classifier
COPY --from=build-classifier \
    /workspace/TensorRT-Vision/build/libengine.so \
    /usr/local/lib/libengine.so
COPY --from=build-classifier \
    /workspace/TensorRT-Vision/build/subprojects/tomlplusplus-3.4.0/src/libtomlplusplus.so.3.4.0 \
    /usr/local/lib/libtomlplusplus.so.3.4.0
COPY --from=build-classifier \
    /workspace/TensorRT-Vision/build/app/classifier/classify \
    /workspace/TensorRT-Vision/build/app/classifier/classify
COPY --from=build-classifier \
    /workspace/TensorRT-Vision/app/classifier/data \
    /workspace/TensorRT-Vision/build/app/classifier/data
RUN ldconfig
WORKDIR /workspace/TensorRT-Vision/build/app/classifier

# ─── Stage 9: segmenter ───────────────────────────────────────────────────────
FROM runtime-base AS segmenter
COPY --from=build-segmenter \
    /workspace/TensorRT-Vision/build/libengine.so \
    /usr/local/lib/libengine.so
COPY --from=build-segmenter \
    /workspace/TensorRT-Vision/build/subprojects/tomlplusplus-3.4.0/src/libtomlplusplus.so.3.4.0 \
    /usr/local/lib/libtomlplusplus.so.3.4.0
COPY --from=build-segmenter \
    /workspace/TensorRT-Vision/build/app/segmenter/segment \
    /workspace/TensorRT-Vision/build/app/segmenter/segment
COPY --from=build-segmenter \
    /workspace/TensorRT-Vision/app/segmenter/data \
    /workspace/TensorRT-Vision/build/app/segmenter/data
RUN ldconfig
WORKDIR /workspace/TensorRT-Vision/build/app/segmenter

# ─── Stage 10: reid ───────────────────────────────────────────────────────────
FROM runtime-base AS reid
COPY --from=build-reid \
    /workspace/TensorRT-Vision/build/libengine.so \
    /usr/local/lib/libengine.so
COPY --from=build-reid \
    /workspace/TensorRT-Vision/build/subprojects/tomlplusplus-3.4.0/src/libtomlplusplus.so.3.4.0 \
    /usr/local/lib/libtomlplusplus.so.3.4.0
COPY --from=build-reid \
    /workspace/TensorRT-Vision/build/app/reid/reid \
    /workspace/TensorRT-Vision/build/app/reid/reid
COPY --from=build-reid \
    /workspace/TensorRT-Vision/app/reid/data \
    /workspace/TensorRT-Vision/build/app/reid/data
# reid export uses torchreid (not in the shared export venv) and its own CLI;
# layer the extra deps onto the baked venv and bake the script to /opt.
COPY --from=build-reid \
    /workspace/TensorRT-Vision/app/reid/torchreid-cli.py \
    /opt/torchreid-cli.py
COPY --from=build-reid \
    /workspace/TensorRT-Vision/app/reid/requirements.txt \
    /tmp/reid-requirements.txt
RUN ${EXPORT_VENV}/bin/pip install --no-cache-dir -r /tmp/reid-requirements.txt \
    && rm /tmp/reid-requirements.txt
RUN ldconfig
WORKDIR /workspace/TensorRT-Vision/build/app/reid

# ─── Stage 11: detector ───────────────────────────────────────────────────────
FROM runtime-base AS detector
COPY --from=build-detector \
    /workspace/TensorRT-Vision/build/libengine.so \
    /usr/local/lib/libengine.so
COPY --from=build-detector \
    /workspace/TensorRT-Vision/build/subprojects/tomlplusplus-3.4.0/src/libtomlplusplus.so.3.4.0 \
    /usr/local/lib/libtomlplusplus.so.3.4.0
COPY --from=build-detector \
    /workspace/TensorRT-Vision/build/app/detector/detect \
    /workspace/TensorRT-Vision/build/app/detector/detect
RUN ldconfig
WORKDIR /workspace/TensorRT-Vision/build/app/detector

# ─── Stage 12: mot ────────────────────────────────────────────────────────────
FROM runtime-base AS mot
COPY --from=build-mot \
    /workspace/TensorRT-Vision/build/libengine.so \
    /usr/local/lib/libengine.so
COPY --from=build-mot \
    /workspace/TensorRT-Vision/build/subprojects/tomlplusplus-3.4.0/src/libtomlplusplus.so.3.4.0 \
    /usr/local/lib/libtomlplusplus.so.3.4.0
COPY --from=build-mot \
    /workspace/TensorRT-Vision/build/subprojects/mot.cpp/libmot.so \
    /usr/local/lib/libmot.so
COPY --from=build-mot \
    /workspace/TensorRT-Vision/build/app/mot/mot \
    /workspace/TensorRT-Vision/build/app/mot/mot
COPY --from=build-mot \
    /workspace/TensorRT-Vision/app/mot/data \
    /workspace/TensorRT-Vision/build/app/mot/data
RUN ldconfig
WORKDIR /workspace/TensorRT-Vision/build/app/mot
