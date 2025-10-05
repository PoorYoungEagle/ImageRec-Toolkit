@echo off

REM Compress a trained model to reduce its size.
REM You can modify the model path or output directory.
REM Use python -m imagerec tools compress --help for more arguments.
REM This example shows compression for a single model.

set MODEL_PATH=examples/example_data/model.pt
set OUTPUT_PATH=examples/outputs

python -m imagerec tools compress --model-path %MODEL_PATH% --output-path %OUTPUT_PATH%