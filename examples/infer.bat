@echo off

REM Perform inference on a single image or a directory of images.
REM You can modify the model path and input path.
REM Use python -m imagerec infer --help for more arguments.
REM Outputs will be printed to the console.

echo For a single image...
set MODEL_PATH=examples/example_data/model.pt
set INPUT_PATH=examples/example_data/val/Colosseum/colo11.jpg

python -m imagerec infer --model-path %MODEL_PATH% --input-path %INPUT_PATH%

echo For a directory of images...
set INPUT_PATH=examples/example_data/val/Colosseum

python -m imagerec infer --model-path %MODEL_PATH% --input-path %INPUT_PATH%