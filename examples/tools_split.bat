@echo off

REM Split a dataset into training and validation sets.
REM You can modify the dataset directory, output directory, or split ratio.
REM Use python -m imagerec tools split --help for more arguments.
REM This will write the split dataset into the examples/outputs directory.

set INPUT_DIR=examples/example_data/train
set OUTPUT_DIR=examples/outputs

python -m imagerec tools split --input-dir %INPUT_DIR% --output-dir %OUTPUT_DIR% --split 0.25