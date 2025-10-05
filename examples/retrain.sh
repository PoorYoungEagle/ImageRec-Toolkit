#!/bin/bash

# Retrain an existing model with new data or additional classes.
# You can modify the dataset directory, model path, or training parameters.
# Use python -m imagerec retrain --help for more arguments.
# This will save the retrained model into the examples/outputs/ directory.

DATA_DIR="examples/example_data" # add in a new class along with the original classes for train and val (you can use the split tool to split the class into train and val and then place it in)
MODEL_PATH="examples/example_data/model.pt" # create a model with train and use that path
OUTPUT_DIR="examples/outputs"
MODEL_NAME="example_model"

python -m imagerec retrain \
    --data-dir $DATA_DIR \
    --model-path $MODEL_PATH \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME \
    --epochs 5 \
    --batch-size 16
