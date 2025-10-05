#!/bin/bash

# Split a dataset into training and validation sets.
# You can modify the dataset directory, output directory, or split ratio.
# Use python -m imagerec tools split --help for more arguments.
# This will write the split dataset into the examples/outputs directory.

INPUT_DIR="examples/example_data/train" # create a model with train and use that path
OUTPUT_DIR="examples/outputs"

python -m imagerec tools split \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --split 0.25