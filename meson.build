project('TensorRT-Vision', 'cpp',
        version : '0.2',
        default_options : [
          'cpp_std=c++23',
          'warning_level=3',
          'werror=true'
        ]
)

# Get vision-core dependency
vision_core_proj = subproject('vision-core')
vision_core_dep = vision_core_proj.get_variable('vision_core_dep')

# Dependencies
cuda_dep = dependency('cuda')
spdlog_dep = dependency('spdlog')
json_dep = dependency('nlohmann_json')
opencv_dep = dependency('opencv4',
  modules: ['core', 'highgui', 'imgproc', 'imgcodecs', 'video', 'videoio'],
  include_type: 'system'
)

_trt_prefix = get_option('tensorrt_prefix')
if _trt_prefix != ''
  tensorrt_include_dir = _trt_prefix / 'include'
  tensorrt_lib_dir     = _trt_prefix / 'lib'
else
  _arch = host_machine.cpu_family() == 'aarch64' ? 'aarch64-linux-gnu' : 'x86_64-linux-gnu'
  tensorrt_include_dir = '/usr/include'
  tensorrt_lib_dir     = '/usr/lib' / _arch
endif

tensorrt_dep = declare_dependency(
  include_directories : [tensorrt_include_dir],
  link_args : ['-L' + tensorrt_lib_dir, '-lnvinfer', '-lnvinfer_plugin', '-lcudart']
)

dependencies = [opencv_dep, spdlog_dep, json_dep, cuda_dep, tensorrt_dep, vision_core_dep]

# Source files
src_files = files(
  'src/engine/engine.cpp',
  'src/tasks/classification.cpp',
  'src/tasks/detection.cpp',
  'src/tasks/reid.cpp',
  'src/tasks/segmentation.cpp',
  'src/arch/yolo/detector.cpp',
  'src/arch/yolo/segmenter.cpp'
)

# Include
inc_dir = [include_directories('include'),  tensorrt_include_dir]

# Build shared library
engine_lib = shared_library(
  'engine', 
  sources : src_files,
  include_directories : inc_dir,
  dependencies : dependencies,
  install : true
)

# Declare dependencies
engine_dep = declare_dependency(
  link_with : engine_lib,
  include_directories : inc_dir,
  dependencies : dependencies
)

# Make engine_dep available to parent projects
meson.override_dependency('engine', engine_dep)

# Build apps
apps = get_option('build_apps')
if apps.length() > 0
    subdir('app')
endif
