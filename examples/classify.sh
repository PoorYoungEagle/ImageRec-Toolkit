#!/bin/bash

# Classify images using a trained model.
# You can modify the model path, input directory, or output directory.
# Use python -m imagerec classify --help for more arguments.
# Classified results will be saved in the output directory.

MODEL_PATH="examples/example_data/model.pt" # create a model with train and use that path
INPUT_DIR="examples/example_data/train/Colosseum"
OUTPUT_DIR="examples/outputs"

python -m imagerec classify \
    --model-path $MODEL_PATH \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR