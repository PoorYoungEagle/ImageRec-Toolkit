#!/bin/bash

# Perform inference on a single image or a directory of images.
# You can modify the model path and input path.
# Use python -m imagerec infer --help for more arguments.
# Outputs will be printed to the console.

MODEL_PATH="examples/example_data/model.pt" # create a model with train and use that path
INPUT_PATH="examples/example_data/val/Colosseum/colo11.jpg"

echo For a single image...
python -m imagerec infer \
    --model-path $MODEL_PATH \
    --input-path $INPUT_PATH

INPUT_PATH="examples/example_data/val/Colosseum"

echo For a directory of images...
python -m imagerec infer \
    --model-path $MODEL_PATH \
    --input-path $INPUT_PATH