@echo off

REM Plot training results or data augmentations for visualization.
REM You can modify the CSV path, image path, or augmentation parameters.
REM Use python -m imagerec plot --help for more arguments.
REM This will save plots or previews based on the chosen mode.

echo For CSV...
set CSV_PATH=examples/example_data/example_log.csv

python -m imagerec plot csv --csv-path %CSV_PATH%

echo For augments...
set IMAGE_PATH=examples/example_data/train/Colosseum/colo1.jpg

python -m imagerec plot augments --image-path %IMAGE_PATH% --num-samples 20 --visual-type "grid"