#!/bin/bash

# Evaluate a trained model on a validation dataset.
# You can modify the dataset directory or model path.
# Use python -m imagerec evaluate --help for more arguments.
# This will print evaluation metrics to the console.

DATA_DIR="examples/example_data/val"
MODEL_PATH="examples/example_data/model.pt"

python -m imagerec evaluate \
    --data-dir $DATA_DIR \
    --model-path $MODEL_PATH