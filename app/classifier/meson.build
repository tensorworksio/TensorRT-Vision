classifier_src = files(
    'main.cpp'
)

# Define data directory
classifier_data_dir = 'data'
build_data_dir = meson.current_build_dir() / 'data'

# Create data directories
run_command('mkdir', '-p', build_data_dir, check: false)
run_command('mkdir', '-p', 
    meson.current_source_dir() / classifier_data_dir, 
    check: false
)

# Create custom target to copy data
custom_target('classifier-data',
    output: 'data',
    command: ['cp', '-r', 
        meson.current_source_dir() / classifier_data_dir,
        meson.current_build_dir()
    ],
    build_by_default: true
)

executable('classifier',
    classifier_src,
    dependencies: [engine_dep],
    include_directories: include_directories('.'),
    install: true
)