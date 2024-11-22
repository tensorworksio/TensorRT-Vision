fs = import('fs')

foreach app : get_option('build_apps')
    if fs.exists(app)
        subdir(app)
    else
        warning('App directory "' + app + '" not found, skipping...')
    endif
endforeach