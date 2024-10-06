# Object Classification

## Compile
```bash
meson setup --wipe build -Dbuild_examples=true
meson compile -C build
```

## Configure
Configure your model settings in `config.json`


## Run
```bash
./build/examples/classification -c examples/classification/config.json -i examples/classification/data/image.jpg
```