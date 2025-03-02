src = files(
    'main.cpp'
)

# Define data directory
data_dir = 'data'
build_data_dir = meson.current_build_dir() / 'data'

# Create data directories
run_command('mkdir', '-p', build_data_dir, check: false)
run_command('mkdir', '-p', 
    meson.current_source_dir() / data_dir, 
    check: false
)

custom_target('yolo-data',
    output: 'data',
    command: ['cp', '-r', 
        meson.current_source_dir() / data_dir,
        meson.current_build_dir()
    ],
    build_by_default: true
)

executable('detect',
    src,
    dependencies: [engine_dep],
    include_directories: include_directories('.'),
    install: true
)