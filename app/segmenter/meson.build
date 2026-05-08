src = files(
    'main.cpp'
)

data_dir = 'data'

run_command('mkdir', '-p', meson.current_source_dir() / data_dir, check: false)
run_command('rm', '-f', meson.current_build_dir() / data_dir, check: true)
run_command('ln', '-s', meson.current_source_dir() / data_dir, meson.current_build_dir() / data_dir, check: true)

argparse_dep = dependency('argparse', required: true)

executable('segment',
    src,
    dependencies: [engine_dep, argparse_dep],
    include_directories: include_directories('.'),
    install: true
)
