src = files(
    'main.cpp'
)

data_dir = 'data'

run_command('mkdir', '-p', meson.current_source_dir() / data_dir, check: false)

custom_target('segmenter-data',
    output: '.segmenter-data-stamp',
    command: [
        'sh', '-c',
        'mkdir -p "$1" && cp -rT "$2" "$1" && touch "$3"',
        'sh',
        meson.current_build_dir() / data_dir,
        meson.current_source_dir() / data_dir,
        '@OUTPUT@'
    ],
    build_by_default: true
)

executable('segment',
    src,
    dependencies: [engine_dep],
    include_directories: include_directories('.'),
    install: true
)
