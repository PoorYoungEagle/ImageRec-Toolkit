@echo off

REM Classify images using a trained model.
REM You can modify the model path, input directory, or output directory.
REM Use python -m imagerec classify --help for more arguments.
REM Classified results will be saved in the output directory.

set MODEL_PATH=examples/example_data/model.pt
set INPUT_DIR=examples/example_data/train/Colosseum
set OUTPUT_DIR=examples/outputs

python -m imagerec classify --model-path %MODEL_PATH% --input-dir %INPUT_DIR% --output-dir %OUTPUT_DIR%