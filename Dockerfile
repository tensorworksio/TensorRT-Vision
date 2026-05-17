# ─── Configuration ────────────────────────────────────────────────────────────
#
# TRT_IMAGE: NGC TensorRT developer image (build stage only).
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

# CUDA_IMAGE: slim CUDA + cuDNN runtime image (final app stages).
# Must match the CUDA *major.minor* version of TRT_IMAGE (patch can differ).
# Smaller variants exist (base, runtime) but cudnn-runtime is the minimum TRT needs.
# Check available tags at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda
#
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# ─── Stage 1: builder ─────────────────────────────────────────────────────────
FROM ${TRT_IMAGE} AS builder

# <print> header requires GCC 14+
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

# Collect runtime .so files that the slim CUDA image doesn't ship:
#   /staging/usr  — OpenCV, spdlog/{fmt}, Qt5, GCC-14 libstdc++, TRT runtime,
#                   and their media/image-format transitive deps.
#   /staging/cuda — NVRTC and cuFFT, which are only in the CUDA devel image.
#
# Symlinks are preserved (-P) and the /usr tree is reconstructed (--parents)
# so COPY --from=builder /staging/usr /usr overlays cleanly in the runtime stage.
RUN mkdir -p /staging/usr /staging/cuda \
    && find /usr \( -type f -o -type l \) \( \
           -name 'libopencv_*.so*'             \
           -o -name 'libspdlog.so*'            \
           -o -name 'libfmt.so*'               \
           -o -name 'libstdc++.so.6*'          \
           -o -name 'libgcc_s.so*'             \
           -o -name 'libpng16.so*'             \
           -o -name 'libjpeg*.so*'             \
           -o -name 'libtiff.so*'              \
           -o -name 'libwebp*.so*'             \
           -o -name 'libavcodec.so*'           \
           -o -name 'libavformat.so*'          \
           -o -name 'libavutil.so*'            \
           -o -name 'libswscale.so*'           \
           -o -name 'libswresample.so*'        \
           -o -name 'libdc1394.so*'            \
           -o -name 'libv4l*.so*'              \
           -o -name 'libQt5Core.so*'           \
           -o -name 'libQt5Gui.so*'            \
           -o -name 'libQt5Widgets.so*'        \
           -o -name 'libQt5XcbQpa.so*'         \
           -o -name 'libQt5DBus.so*'           \
           -o -name 'libdouble-conversion.so*' \
           -o -name 'libzstd.so*'              \
           -o -name 'libicui18n.so*'           \
           -o -name 'libicuuc.so*'             \
           -o -name 'libicudata.so*'           \
           -o -name 'libnvinfer*.so*'          \
           -o -name 'libnvonnxparsers*.so*'    \
           -o -name 'libnvparsers*.so*'        \
       \) -exec cp -P --parents -t /staging/usr {} + \
    && find /usr -path '*/qt5/plugins' -type d \
           -exec sh -c 'cp -rP --parents "$1" /staging/usr/' _ {} \; 2>/dev/null \
    ; true \
    && find /usr/local/cuda/lib64 \( -type f -o -type l \) \( \
           -name 'libnvrtc*.so*'  \
           -o -name 'libcufft*.so*' \
       \) -exec cp -P {} /staging/cuda/ \;

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
# Slim CUDA + cuDNN runtime image (~2-3 GB) instead of the full TRT developer
# image (~10 GB).  Everything the apps need at inference time arrives via the
# staged .so files from the builder.
FROM ${CUDA_IMAGE} AS runtime-base

COPY --from=builder /staging/usr /usr
# NVRTC and cuFFT are only in the CUDA devel image, not in cudnn-runtime.
# TRT uses NVRTC for JIT kernel compilation at engine build time.
COPY --from=builder /staging/cuda/ /usr/local/cuda/lib64/
# trtexec is in the TRT developer image only; copy the binary so users can
# convert ONNX models to TRT engines without a local TensorRT install.
COPY --from=builder /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec
RUN ldconfig

# OpenCV's system-level transitive dependencies (image formats, codecs, display).
# These are not covered by the staged .so files because they are system packages,
# not libraries OpenCV directly ships.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libopengl0 \
    libqt5opengl5t64 libqt5test5t64 \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    libgphoto2-6t64 libgphoto2-port12t64 libraw1394-11 libusb-1.0-0 \
    libblas3 liblapack3 libtbb12 \
    libprotobuf32t64 \
    libopenjp2-7 libopenexr-3-1-30 \
    libaom3 libdav1d7 libmp3lame0 libopus0 libsnappy1v5 libspeex1 \
    libtheora0 libvorbis0a libvorbisenc2 libvpx9 libxvidcore4 \
    libx264-164 libx265-199 \
    libva2 libva-drm2 libva-x11-2 libvdpau1 libvpl2 \
    libcairo2 libdeflate0 libharfbuzz0b libjbig0 liblerc4 \
    libmd4c0 libpcre2-16-0 librsvg2-2 libsharpyuv0 libsoxr0 \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

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
    /workspace/TensorRT-Vision/build/app/mot/mot \
    /workspace/TensorRT-Vision/build/app/mot/mot
COPY --from=build-mot \
    /workspace/TensorRT-Vision/app/mot/data \
    /workspace/TensorRT-Vision/build/app/mot/data
RUN ldconfig
WORKDIR /workspace/TensorRT-Vision/build/app/mot
