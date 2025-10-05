# ImageRec-Toolkit
A PyTorch based CLI toolkit for image classification, augmentation, inference, and more. It lets you easily train and retrain models on custom datasets, evaluate model performance, and perform inference or classification on single images or folders.

![](/images/image10.png)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)  
- [Usage](#usage)  
  - [Train](#train)
  - [Retrain](#retrain)
  - [Inference](#inference)
  - [Classification](#classification)
  - [Evaluation](#evaluation)
  - [Plot](#plot)
    - [Plot CSV](#csv)
    - [Plot Augmentations](#augments)
  - [Tools](#tools)
    - [Split Dataset](#split-dataset)
    - [Limit Class Images](#limit-class-images)
    - [Compress Model](#compress-model)
    - [Export Model](#export-model)
    - [Find Learning Rate](#find-learning-rate)
- [Testing](#testing)
- [Sources](#sources)

## Overview
ImageRec is a simple Python CLI toolkit for image classification, augmentation, inference, and more. One can quickly test models, experiment with datasets, and handle image data without writing any custom code. It contains these commands:
- Train and retrain models on custom datasets
- Run inference to predict image classes
- Classify images into their respective folders
- Evaluate model performance on validation data
- Plot training logs and visualize image augmentations
- Use various tools for dataset or model manipulation

## Installation
Clone the repository :
```bash
git clone https://github.com/PoorYoungEagle/ImageRec-Toolkit.git
cd ImageRec-Toolkit
pip install -r requirements.txt
```

### CUDA Support
By default, PyTorch will install the CPU version. If you have an NVIDIA GPU and want faster training and inference, you should install the CUDA version of PyTorch instead:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
You can check other CUDA versions and installation options at the [PyTorch Website](https://pytorch.org/get-started/locally)

## Usage
It requires the -m module syntax to run. Use the following command to see all available options and commands:
```bash
python -m imagerec --help
```
For a full list of available options for each command, use the following example:
```bash
python -m imagerec train --help
```
Check the examples/ directory for sample scripts and usage examples.

### Train
Train a new model on a custom dataset and save it.
```bash
python -m imagerec train \
    --data-dir "examples/example_data" \
    --output-dir "examples/outputs" \
    --model-name "example_model" \
    --epochs 5 \
    --batch-size 16
```
![](/images/image4.png)
![](/images/image1.png)
### Retrain
Retrain an existing model with same or new data.
```bash
python -m imagerec retrain \
    --data-dir "examples/example_data" \
    --model-path "examples/example_data/model.pt" \
    --output-dir "examples/outputs" \
    --model-name "example_model" \
    --new-classes "Sphinx, Petra"
    --epochs 5 \
    --batch-size 16
```

### Inference
Infer on a single image or an entire folder of images to predict their classes.
```bash
python -m imagerec infer \
    --model-path "examples/example_data/model.pt" \
    --input-path "examples/example_data/Colosseum/colo.png"
```

### Classification
Classify a folder of images into their respective classes and organize them into folders based on the predicted labels.
```bash
python -m imagerec classify \
    --model-path "examples/example_data/model.pt" \
    --input-dir "examples/example_data/Colosseum" \
    --output-dir "examples/outputs"
```
![](/images/image7.png)
### Evaluation
Evaluate the model's accuracy and other metrics using the provided dataset and displays the results in the terminal.
```bash
python -m imagerec evaluate \
    --data-dir "examples/example_data/test" \
    --model-path "examples/example_data/model.pt"
```
![](/images/image5.png)
### Plot
Plot training logs or visualize image augmentations to better understand model performance and data transformations.

#### CSV
```bash
python -m imagerec plot csv \
    --csv-path "examples/example_data/example_log.csv"
```
![](/images/image2.png)
#### Augments
```bash
python -m imagerec plot augments \
    --image-path "examples/example_data/Colosseum/colo.png" \
    --num-samples 20
```
![](/images/image3.png)
### Tools
ImageRec includes a set of utility commands to manipulate datasets and models.

#### Split Dataset
Split a dataset into training and validation sets.
```bash
python -m imagerec tools split \
    --input-dir "examples/example_data/classes" \
    --output-dir "examples/outputs" \
    --split 0.25
```

#### Limit Class Images
Limit the number of images per class in the dataset.
```bash
python -m imagerec tools limit \
    --input-dir "examples/example_data/class" \
    --output-dir "examples/outputs" \
    --max-images 50
```

#### Compress Model
Compress a trained model to reduce its size.
```bash
python -m imagerec tools compress \
    --model-path "examples/example_data/model.pt" \
    --output-path "examples/outputs"
```
![](/images/image8.png)
#### Export Model
Export a trained model into different formats (ONNX and TorchScript).
```bash
python -m imagerec tools export \
    --format "onnx" or "torchscript" \
    --model-path "examples/example_data/model.pt" \
    --output-path "examples/outputs"
```

#### Find Learning Rate
Find suitable learning rate on a dataset.
```bash
python -m imagerec tools find-lr \
    --data-dir "examples/example_data/classes" \
    --architecture "resnet34"
```
![](/images/image6.png)
## Testing
ImageRec includes tests to make sure that all commands work correctly.
You can run the tests using pytest.
```bash
pip install pytest
pytest tests/
```
![](/images/image9.PNG)
> Running them is optional but recommended if you make changes or want to verify your setup.

> Takes around 5 minutes with CUDA.

## Sources
- Image datasets used for examples: https://www.pexels.com
- Learning rate methodology: ['Cyclical Learning Rates for Training Neural Networks'](https://arxiv.org/abs/1506.01186)

Any feedback is welcome!
