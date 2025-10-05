@echo off

REM Limit the number of images per class in the dataset.
REM You can modify the dataset directory, output directory, or maximum number of images.
REM Use python -m imagerec tools limit --help for more arguments.
REM This example runs in dry-run mode and does not modify files, only shows what would happen.

set INPUT_DIR=examples/example_data/train
set OUTPUT_DIR=examples/outputs

python -m imagerec tools limit --input-dir %INPUT_DIR% --output-dir %OUTPUT_DIR% --max-images 5 --dry-run