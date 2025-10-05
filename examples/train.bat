@echo off

REM Train a model on the example dataset of landmarks.
REM You can modify the dataset directory, architecture, or training parameters.
REM Use python -m imagerec train --help for more arguments.
REM This will save the trained model into the examples/outputs/ directory.

set DATA_DIR=examples/example_data
set OUTPUT_DIR=examples/outputs
set MODEL_NAME=example_model
set ARCHITECTURE=shufflenet_v2_x0_5

python -m imagerec train --data-dir %DATA_DIR% --output-dir %OUTPUT_DIR% --model-name %MODEL_NAME% --architecture %ARCHITECTURE% --epochs 5 --batch-size 16