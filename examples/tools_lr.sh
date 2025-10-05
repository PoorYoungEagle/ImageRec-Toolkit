#!/bin/bash

# Find a suitable learning rate on the example dataset of landmarks.
# You can modify the dataset directory or model architecture.
# Use python -m imagerec tools find-lr --help for more arguments.
# This will display a plot showing loss vs. learning rate.

DATA_DIR="examples/example_data/train"
ARCHITECTURE="shufflenet_v2_x0_5"

python -m imagerec tools find-lr \
    --data-dir $DATA_DIR \
    --architecture $ARCHITECTURE