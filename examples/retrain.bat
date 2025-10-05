@echo off

REM Retrain an existing model with new data or additional classes.
REM You can modify the dataset directory, model path, or training parameters.
REM Use python -m imagerec retrain --help for more arguments.
REM This will save the retrained model into the examples/outputs/ directory.

set DATA_DIR=examples/example_data
set MODEL_PATH=examples/example_data/model.pt
set OUTPUT_DIR=examples/outputs
set MODEL_NAME=example_model

python -m imagerec retrain --data-dir %DATA_DIR% --model-path %MODEL_PATH% --output-dir %OUTPUT_DIR% --model-name %MODEL_NAME% --epochs 5 --batch-size 16