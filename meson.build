project('TensorRT-Vision', 'cpp',
        version : '0.2',
        default_options : ['warning_level=3', 'cpp_std=c++17', 'debug=true', 'optimization=0'])

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

dlib_dep = dependency('dlib-1', method : 'pkg-config')

# Source files
engine_src = files(
  'src/engine/engine.cpp',
  'src/engine/classifier.cpp',
  'src/engine/reid.cpp',
  'src/engine/yolo.cpp',

  'src/tracking/tracker.cpp',
  'src/tracking/sort.cpp',
  'src/tracking/botsort.cpp',

  'src/tracking/kalman/xysr.cpp',
  'src/tracking/kalman/xywh.cpp'
)

# Include 
engine_inc = [include_directories('include'),  tensorrt_include_dir]

# Build shared library
engine_lib = shared_library(
  'engine', 
  sources : engine_src,
  include_directories : engine_inc,
  dependencies : [boost_dep, opencv_dep, spdlog_dep, json_dep, cuda_dep, tensorrt_dep, dlib_dep],
  install : true
)

# Declare dependencies
engine_dep = declare_dependency(
  link_with : engine_lib,
  include_directories : engine_inc,
  dependencies : [boost_dep, opencv_dep, spdlog_dep, json_dep, cuda_dep, tensorrt_dep, dlib_dep]
)

# Make engine_dep available to parent projects
meson.override_dependency('engine', engine_dep)

# Build apps
apps = get_option('build_apps')
if apps.length() > 0
    subdir('app')
endif
