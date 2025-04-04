project('TensorRT-Vision', 'cpp',
        version : '0.2',
        default_options : [
          'cpp_std=c++17',
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
boost_dep = dependency('boost', 
  modules: ['filesystem', 'program_options', 'json']
)
opencv_dep = dependency('opencv4',
  modules: ['core', 'highgui', 'imgproc', 'imgcodecs', 'video', 'videoio']
)

tensorrt_include_dir = '/usr/include'
tensorrt_lib_dir = '/usr/lib/x86_64-linux-gnu'

tensorrt_dep = declare_dependency(
  include_directories : [tensorrt_include_dir],
  link_args : ['-L' + tensorrt_lib_dir, '-lnvinfer', '-lnvinfer_plugin', '-lcudart']
)

dependencies = [boost_dep, opencv_dep, spdlog_dep, json_dep, cuda_dep, tensorrt_dep, vision_core_dep]

# Source files
src_files = files(
  'src/engine/engine.cpp',
  'src/models/classification/classifier.cpp',
  'src/models/detection/yolo.cpp',
  'src/models/reid/reid.cpp',
  'src/models/segmentation/yolo.cpp'
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
