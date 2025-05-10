# OCR

## Overview
Optical Character Recognition using TensorRT.

## Export Model
1. Prepare Python env:
```shell
mkdir data
python3 -m venv venv
./venv/bin/pip3 install -r requirement.txt
```

2. Export Detection model to ONNX
Get a text detection model [here](https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/model_list.html)
```shell
./venv/bin/paddle2onnx \
--model_dir ./data/en_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ppocrv3_det.onnx \
--opset_version 17
```

3. Export PaddleOCR model to ONNX
Get a text recognizer model [here](https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/model_list.html)
```shell
./venv/bin/paddle2onnx \
--model_dir ./data/en_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ppocrv3_rec.onnx \
--opset_version 17
```

4. Convert to TensorRT engine:
```shell
trtexec --onnx=data/ppocrv3_det.onnx --saveEngine=data/ppocrv3_det.engine --fp16
trtexec --onnx=data/ppocrv3_rec.onnx --saveEngine=data/ppocrv3_rec.engine --fp16
```

## Configure
In `data` folder, add your `config.json`:
```json
{
    "detector": {
        "model_path": "./data/ppocrv3_det.engine",
        "batch_size": 1,
        "precision": 16
    },
    "recognizer": {
        "model_path": "./data/ppocrv3_rec.engine",
        "batch_size": 1,
        "precision": 16
    },
    "mask_threshold": 0.5,
    "top_k": 1000,
    "min_area": 100,
    "vocabulary": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "{",
        "|",
        "}",
        "~",
        "!",
        "\"",
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        " "
    ],
    "confidence_threshold": 0.1
}
```

## Compile
```shell
# in root directory
meson setup build -Dbuild_apps=ocr
meson compile -C build
```

## Run

### Display
```shell
# in root directory
cd build/app/ocr
./ocr -i data/example.png -c data/config.json -d
```