# Add mot.cpp as subproject
mot_proj = subproject('mot.cpp', required: true)
if not mot_proj.found()
    error('mot.cpp subproject not found. Please make sure it exists in subprojects directory.')
endif
mot_dep = mot_proj.get_variable('mot_dep')

src_files = files(
    'main.cpp'
)

data_dir = 'data'

run_command('mkdir', '-p', meson.current_source_dir() / data_dir, check: false)

custom_target('mot-data',
    output: '.mot-data-stamp',
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

executable('mot',
    src_files,
    dependencies: [engine_dep, mot_dep],
    include_directories: include_directories('.'),
    install: true
)
