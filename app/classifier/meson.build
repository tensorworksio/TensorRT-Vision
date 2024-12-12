classifier_src = files(
    'main.cpp'
)

# Define data directory
classifier_data_dir = 'data'
build_data_dir = meson.current_build_dir() / 'data'

# Create custom target to copy data
run_command('mkdir', '-p', build_data_dir, check: false)

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