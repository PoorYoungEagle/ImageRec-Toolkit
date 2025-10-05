#!/bin/bash

# Plot training results or data augmentations for visualization.
# You can modify the CSV path, image path, or augmentation parameters.
# Use python -m imagerec plot --help for more arguments.
# This will save plots or previews based on the chosen mode.

CSV_PATH="examples/example_data/example_log.csv"

echo For CSV...
python -m imagerec plot csv \
    --csv-path $CSV_PATH

IMAGE_PATH="examples/example_data/train/Colosseum/colo1.jpg"

echo For augments...
python -m imagerec plot augments \
    --image-path $IMAGE_PATH \
    --num-samples 20