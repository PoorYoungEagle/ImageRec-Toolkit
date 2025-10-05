#!/bin/bash

# Export a trained model into different formats (ONNX and TorchScript).
# You can modify the model path, output directory, or format.
# Use python -m imagerec tools export --help for more arguments.
# This example first exports to ONNX, then to TorchScript.

MODEL_PATH="examples/example_data/model.pt" # create a model with train and use that path
OUTPUT_PATH="examples/outputs"
FORMAT="onnx"

echo For ONNX...
python -m imagerec tools export \
    --format $FORMAT \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_PATH

INPUT_PATH="examples/example_data/val/Colosseum"

FORMAT="torchscript"

echo For TorchScript...
python -m imagerec tools export \
    --format $FORMAT \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_PATH