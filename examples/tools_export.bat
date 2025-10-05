@echo off

REM Export a trained model into different formats (ONNX and TorchScript).
REM You can modify the model path, output directory, or format.
REM Use python -m imagerec tools export --help for more arguments.
REM This example first exports to ONNX, then to TorchScript.

set MODEL_PATH=examples/example_data/model.pt
set OUTPUT_PATH=examples/outputs
set FORMAT=onnx

python -m imagerec tools export --format %FORMAT% --model-path %MODEL_PATH% --output-path %OUTPUT_PATH%

set FORMAT=torchscript

python -m imagerec tools export --format %FORMAT% --model-path %MODEL_PATH% --output-path %OUTPUT_PATH%