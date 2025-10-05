#!/bin/bash

# Compress a trained model to reduce its size.
# You can modify the model path or output directory.
# Use python -m imagerec tools compress --help for more arguments.
# This example shows compression for a single model.

MODEL_PATH="examples/example_data/model.pt" # create a model with train and use that path
OUTPUT_PATH="examples/outputs"

python -m imagerec tools compress \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_PATH