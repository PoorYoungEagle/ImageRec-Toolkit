#!/bin/bash

# Train a model on the example dataset of landmarks.
# You can modify the dataset directory, architecture, or training parameters.
# Use python -m imagerec train --help for more arguments.
# This will save the trained model into the examples/outputs/ directory.

DATA_DIR="examples/example_data"
OUTPUT_DIR="examples/outputs"
MODEL_NAME="example_model"
ARCHITECTURE="shufflenet_v2_x0_5"

python -m imagerec train \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME \
    --architecture $ARCHITECTURE \
    --epochs 5 \
    --batch-size 16
