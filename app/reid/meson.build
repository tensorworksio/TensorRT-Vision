reid_src = files(
    'reid.cpp',
    'main.cpp'
)

# Define data directory
reid_data_dir = 'data'
build_data_dir = meson.current_build_dir() / 'data'

# Create custom target to copy data
run_command('mkdir', '-p', build_data_dir, check: false)

custom_target('reid-data',
    output: 'data',
    command: ['cp', '-r', 
        meson.current_source_dir() / reid_data_dir,
        meson.current_build_dir()
    ],
    build_by_default: true
)

executable('reid',
    reid_src,
    dependencies: [engine_dep],
    include_directories: include_directories('.'),
    install: true
)