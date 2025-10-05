@echo off

REM Find a suitable learning rate on the example dataset of landmarks.
REM You can modify the dataset directory or model architecture.
REM Use python -m imagerec tools find-lr --help for more arguments.
REM This will display a plot showing loss vs. learning rate.

set DATA_DIR=examples/example_data/train
set ARCHITECTURE=shufflenet_v2_x0_5

python -m imagerec tools find-lr --data-dir %DATA_DIR% --architecture %ARCHITECTURE%