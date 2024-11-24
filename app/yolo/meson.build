yolo_src = files(
    'yolo.cpp',
    'main.cpp'
)

# Define data directory
yolo_data_dir = 'data'
build_data_dir = meson.current_build_dir() / 'data'

# Create custom target to copy data
run_command('mkdir', '-p', build_data_dir, check: false)

custom_target('yolo-data',
    output: 'data',
    command: ['cp', '-r', 
        meson.current_source_dir() / yolo_data_dir,
        meson.current_build_dir()
    ],
    build_by_default: true
)

executable('yolo',
    yolo_src,
    dependencies: [engine_dep],
    include_directories: include_directories('.'),
    install: true
)