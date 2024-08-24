project('engine.trt', 'cpp')

# Include directory
inc_dir = include_directories('include')

# Dependencies
cuda_dep = dependency('cuda', required : true)
spdlog_dep = dependency('spdlog', required : true)
boost_dep = dependency('boost', modules: ['filesystem', 'program_options', 'json'])
opencv_dep = dependency('opencv4',
  modules: ['core', 'highgui', 'imgproc', 'imgcodecs', 'video', 'videoio']
)

tensorrt_include_dir = '/usr/include'
tensorrt_lib_dir = '/usr/lib/x86_64-linux-gnu'

tensorrt_dep = declare_dependency(
  include_directories : [tensorrt_include_dir],
  link_args : ['-L' + tensorrt_lib_dir, '-lnvinfer', '-lnvinfer_plugin', '-lcudart']
)

# Source files
src_files = files(
  'src/engine.cpp',
)

# Build shared library
library(
  'engine', 
  src_files,
  dependencies: [boost_dep, opencv_dep, spdlog_dep, cuda_dep, tensorrt_dep],
  include_directories: [inc_dir, tensorrt_include_dir]
)