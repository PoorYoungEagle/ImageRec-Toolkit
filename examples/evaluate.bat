@echo off

REM Evaluate a trained model on a validation dataset.
REM You can modify the dataset directory or model path.
REM Use python -m imagerec evaluate --help for more arguments.
REM This will print evaluation metrics to the console.

set DATA_DIR=examples/example_data/val
set MODEL_PATH=examples/example_data/model.pt

python -m imagerec evaluate --data-dir %DATA_DIR% --model-path %MODEL_PATH%