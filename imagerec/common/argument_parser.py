import argparse
import yaml
import os
import pathlib
from typing import Any, Dict, Optional
import logging
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for CLI arguments and YAML configs"""

    DEFAULTS = {
        "seed" : 42,
        "device" : "auto",
        "batch_size" : 16,
        "num_workers" : 4,
        "image_size" : 224,
        "lr" : 0.001,
        "epochs" : 50,
        "percentage" : 100,
        "criterion_type" : "crossentropy",
        "scheduler_type" : "step",
        "optimizer_type" : "adam",
        "strategy" : "full"
    }

    @staticmethod
    def load_config(path: str) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        if not os.path.exists(path):
            logger.warning(f"YAML file not found at {path}\nGoing back to default")
            return None
        elif os.path.isdir(path):
            logger.warning(f"Chosen path is a directory: {path}\nGoing back to default")
            return None
        with open(path, 'r') as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML at {path}: {e}")
                return None
        
    @classmethod
    def config_checks(cls, arguments: argparse.Namespace) -> None:
        """Apply YAML config values where CLI args are not provided."""
        config_path = pathlib.Path(arguments.data_config)
        if not config_path.is_absolute():
            script_dir = pathlib.Path(__file__).resolve().parent.parent
            config_path = script_dir / config_path

        arguments.data_config = str(config_path)

        if config_path.exists():
            logger.info(f"Loading data config: {config_path}")
            data = cls.load_config(config_path)
        else:
            logger.warning(f"Config file for data not detected: {arguments.data_config}\nUsing default hard coded values for missing values.")
            data = {}

        for key in cls.DEFAULTS:
            cli_value = getattr(arguments, key, None)
            config_value = data.get(key, None)
            
            if cli_value is not None:
                logger.debug(f"Using CLI argument for {key}: {cli_value}")
                continue
            elif config_value is not None:
                logger.debug(f"Using config file value for {key}: {config_value}")
                setattr(arguments, key, config_value)
            else:
                logger.warning(f"Value for {key} not found. Using default hard coded values.")
                setattr(arguments, key, cls.DEFAULTS[key])    
    
class ArgumentErrors:
    """Error handling logic for CLI arguments"""

    @staticmethod
    def error_checks(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        """Checks and raises errors for invalid values"""

        errors = [
            ArgumentErrors._check_seed,
            ArgumentErrors._check_percentage,
            ArgumentErrors._check_epochs,
            ArgumentErrors._check_batch_size,
            ArgumentErrors._check_image_size,
            ArgumentErrors._check_workers,
            ArgumentErrors._check_learning_rate,
            ArgumentErrors._check_top_n,
            ArgumentErrors._check_class_filters,
            ArgumentErrors._check_confidence_threshold,
            ArgumentErrors._check_max_images
        ]

        for err in errors:
            err(arguments, parser)

    @staticmethod
    def _check_seed(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "seed") and arguments.seed is not None and arguments.seed < 0:
            parser.error("Seed must be a non-negative integer")

    @staticmethod
    def _check_percentage(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "percentage") and not (0 <= arguments.percentage <= 100):
            parser.error("Percentage must be between 1 and 100")

    @staticmethod
    def _check_epochs(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "epochs") and arguments.epochs <= 0:
            parser.error("Number of epochs must be a positive integer")

    @staticmethod
    def _check_batch_size(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "batch_size") and arguments.batch_size <= 0:
            parser.error("Batch size must be a positive integer")

    @staticmethod
    def _check_image_size(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "image_size") and arguments.image_size <= 0:
            parser.error("Image size must be a positive integer")

    @staticmethod
    def _check_workers(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "num_workers") and arguments.num_workers < 0:
            parser.error("Number of workers must be a non-negative integer")

    @staticmethod
    def _check_learning_rate(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "lr") and arguments.lr <= 0:
            parser.error("Learning rate must be a positive number")

    @staticmethod
    def _check_top_n(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "top_n") and arguments.top_n <= 0:
            parser.error("Top predictions must be a positive integer")

    @staticmethod
    def _check_class_filters(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if (hasattr(arguments, "include_classes") and hasattr(arguments, "exclude_classes") 
            and arguments.include_classes and arguments.exclude_classes):
            parser.error("Cannot use both --include-classes and --exclude-classes simultaneously")

    @staticmethod
    def _check_confidence_threshold(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "confidence_threshold") and not (0.0 <= arguments.confidence_threshold <= 1.0):
            parser.error("--confidence-threshold must be between 0.0 and 1.0")

    @staticmethod
    def _check_max_images(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        if hasattr(arguments, "max_images") and arguments.max_images < 0:
            parser.error("Max images must be a positive integer.")

class ArgumentParsers:
    """Class for creating argument parsers"""

    @staticmethod
    def add_common_arguments(parser: argparse.ArgumentParser) -> None:
        """Add arguments common to train/retrain commands."""

        parser.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="Path to data directory containing train/ and validation/ folders."
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed (default: 42)"
        )
        parser.add_argument(
            "--device",
            type=str,
            choices=["auto", "cuda", "cpu"],
            help="Computation device:\n'auto': use CUDA if available, else CPU\n'cuda': force GPU\n'cpu': force CPU. (default: auto)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Number of samples per mini-batch during training. Larger values use more memory. (default: 16)"
        )
        parser.add_argument(
            "--criterion-type",
            type=str,
            choices=["crossentropy", "bce"],
            help="Loss function for training:\n'crossentropy': use CrossEntropyLoss for multi-class classification\n 'bce': use BCEWithLogitsLoss for binary classification tasks. Only works with a single class. (default: crossentropy)"
        )
        parser.add_argument(
            "--scheduler-type",
            type=str,
            choices=["step", "cosine", "plateau", "none"],
            help="Learning rate scheduling strategy:\n'step': decay LR by fixed factor at specified intervals\n'cosine': cosine annealing schedule\n'plateau': reduce LR when validation loss plateaus\n'none': constant learning rate. (default: step)"
        )
        parser.add_argument(
            "--optimizer-type",
            type=str,
            choices=["adam", "adamw", "sgd"],
            help="Optimization algorithm:\n'adam': Adam optimizer with adaptive learning rates\n'adamw': Adam with decoupled weight decay\n'sgd': Stochastic Gradient Descent with momentum. (default: adam)"
        )
        parser.add_argument(
            "--strategy",
            type=str,
            choices=["full", "freeze_backbone", "differential_lr"],
            help="Training strategy:\n'full': train all model parameters\n'freeze_backbone': freeze feature extractor and train only classifier head\n'differential_lr': use different learning rates for backbone vs. head layers. (default: full)"
        )
        parser.add_argument(
            "--lr",
            type=float,
            help="Initial learning rate for the optimizer. (default: 0.001)"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            help="Epochs: Maximum number of complete passes through the training dataset (default: 50)"
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            help="Number of subprocesses for data loading. Higher values can speed up data loading but consume more system memory and CPU. Set to 0 to disable multiprocessing. (default: 4)"
        )
        parser.add_argument(
            "--percentage",
            type=int,
            help="Percentage of available images per class to use during training (1-100). 100 uses all available data. (default: 100)"
        )
        parser.add_argument(
            "--log-dir",
            type=str,
            default="logs",
            help="Directory path where training logs, metrics, and checkpoints will be saved. Will be created if it doesn't exist. (default: logs)"
        )
        parser.add_argument(
            "--data-config",
            type=str,
            default=r"configs\data\default.yaml",
            help="Path to YAML configuration file containing data-specific settings. (default: imagerec/configs/data/default.yaml)"
        )
        parser.add_argument(
            "--transforms-config",
            type=str,
            default=r"configs\transforms\default.yaml",
            help="Path to the transforms config YAML. (default: imagerec/configs/transforms/default.yaml)"
        )
        parser.add_argument(
            "--no-augments",
            dest="use_augments",
            action="store_false",
            help="Disable data augmentation during training."
        )
        parser.set_defaults(use_augments=True)

    @staticmethod
    def parse_train_arguments(subparsers) -> argparse.ArgumentParser:
        """Create parser for train command."""
        train_parser = subparsers.add_parser("train", help="Train a new model")

        ArgumentParsers.add_common_arguments(train_parser)
        train_parser.add_argument(
            "--output-dir",
            type=str,
            default="models",
            help="Directory path where trained models will be saved. The directory will be created if it doesn't exist. (default: models)"
        )
        train_parser.add_argument(
            "--model-name",
            type=str,
            default="model",
            help="Base name for saved model files. The actual filename will include this name along with timestamp. (default: model)"
        )
        train_parser.add_argument(
            "--image-size",
            type=int,
            default=224,
            help="Target size (height and width) in pixels for input images. Larger sizes may improve accuracy but require more GPU memory. (default: 224)"
        )
        train_parser.add_argument(
            "--architecture",
            type=str,
            default="resnet34",
            choices=[
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
                'wide_resnet50_2', 'wide_resnet101_2',
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                'densenet121', 'densenet161', 'densenet169', 'densenet201',
                'alexnet',
                'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'
            ],
            help="Neural network architecture for the model backbone (default: resnet34): %(choices)s"
        )
        train_parser.add_argument(
            "--no-pretrained",
            dest="pretrained",
            action="store_false",
            help="Disable loading of pretrained weights from ImageNet."
        )
        train_parser.set_defaults(pretrained=True)
        train_parser.add_argument(
            "--no-timestamp",
            dest="add_timestamp",
            action="store_false",
            help="Disable adding timestamp to output folder names."
        )
        train_parser.set_defaults(add_timestamp=True)
        return train_parser

    @staticmethod
    def parse_retrain_arguments(subparsers) -> argparse.ArgumentParser:
        """Create parser for retrain command."""
        retrain_parser = subparsers.add_parser("retrain", help="Retrain an existing model")

        ArgumentParsers.add_common_arguments(retrain_parser)
        retrain_parser.add_argument(
            "--output-dir",
            type=str,
            help="Directory path where the retrained model will be saved. If not specified, the retrained model will overwrite the original model in its current location. WARNING: This will permanently replace the original model file. (default: original model directory)"
        )
        retrain_parser.add_argument(
            "--model-name",
            type=str,
            help="Name for the retrained model file. If not specified, uses the original model's filename. (default: original model name)"
        )    
        retrain_parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the existing trained model file to be retrained. Must be a valid model file that was previously saved by this framework."
        )
        retrain_parser.add_argument(
            "--new-classes",
            type=lambda s: [cls.strip() for cls in s.split(',')],
            help="Comma-separated list of new class names to include in retraining. These classes will then be added to the model and trained. Example: --new-classes Parrot,Eagle (default: none)"
        )
        retrain_parser.add_argument(
            "--skip-missing-classes",
            dest="skip_missing_classes",
            action="store_true",
            help="If False, raises an error when there are missing classes. If True, it skips the missing classes and doesn't include them."
        )
        return retrain_parser

    @staticmethod
    def parse_inference_arguments(subparsers) -> argparse.ArgumentParser:
        """Create parser for inference command."""
        infer_parser = subparsers.add_parser("infer", help="Infer images from using a model")

        infer_parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the trained model file for inference. Supports various formats including PyTorch (.pth, .pt), TorchScript and ONNX (.onnx)."
        )
        infer_parser.add_argument(
            "--input-path",
            type=str,
            required=True,
            help="Path to input image file or directory containing images for inference. Supported formats: JPEG, JPG, PNG."
        )
        infer_parser.add_argument(
            "--config-labels",
            type=str,
            help="Path to configuration file containing class labels for TorchScript or ONNX models. Required when using TorchScript or ONNX format models that don't have embedded class information. Should be a JSON file which would be created when using export tool."
        )
        infer_parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cuda", "cpu"],
            help="Computation device:\n'auto': use CUDA if available, else CPU\n'cuda': force GPU\n'cpu': force CPU. (default: auto)"
        )
        infer_parser.add_argument(
            "--top-n",
            type=int,
            default=3,
            help="Number of top-confidence predictions to display for each image. (default: 3)"
        )
        infer_parser.add_argument(
            "--include-classes",
            type=lambda s: [cls.strip() for cls in s.split(',')],
            help="Comma-separated list of classes to include for inference. Ex: 'Dog, Cat, Parrot'"
        )

        infer_parser.add_argument(
            "--exclude-classes",
            type=lambda s: [cls.strip() for cls in s.split(',')],
            help="Comma-separated list of classes to exclude from inference. Ex: 'Dog, Cat, Parrot'"
        )
        return infer_parser

    @staticmethod
    def parse_classify_parser(subparsers) -> argparse.ArgumentParser:
        """Create parser for classify command."""
        classify_parser = subparsers.add_parser("classify", help="Classify images into inferred folders")

        classify_parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the trained model file for classification."
        )
        classify_parser.add_argument(
            "--input-dir",
            type=str,
            required=True,
            help="Path to input directory containing images to classify. All images will be processed and moved/copied to class-specific output directories based on model predictions."
        )
        classify_parser.add_argument(
            "--output-dir",
            type=str,
            help="Root directory where classified images will be organized. Creates subdirectories for each predicted class and moves/copies images accordingly. (default: stores in original input path)"
        )
        classify_parser.add_argument(
            "--config-labels",
            type=str,
            help="Path to configuration file containing class labels for ONNX models. Required when using ONNX format models that don't have embedded class information. Should be a JSON file which would be created when using export tool."
        )
        classify_parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cuda", "cpu"],
            help="Computation device:\n'auto': use CUDA if available, else CPU\n'cuda': force GPU\n'cpu': force CPU. (default: auto)"
        )
        classify_parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum confidence/probability required for a prediction to be considered valid (0.0 - 1.0). Will create a directory called 'Uncertain' if image below confidence threshold. (default: 0.5)"
        )
        classify_parser.add_argument(
            "--delete-original",
            dest="delete_original",
            action="store_true",
            help="Deletes original images after classifying instead of copying them. WARNING: This permanently removes the original files. Use with caution and ensure you have backups. Default behavior is to copy files, preserving originals but requires extra space."
        )
        return classify_parser

    @staticmethod
    def parse_evaluation_arguments(subparsers) -> argparse.ArgumentParser:
        """Create parser for evaluate command."""
        evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model metrics")
        
        evaluate_parser.add_argument(
            "--data-dir",
            type=str,
            help="Path to data directory containing classes folders."
        )
        evaluate_parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the trained model file to evaluate."
        )
        evaluate_parser.add_argument(
            "--type",
            type=str,
            dest="evaluate_type",
            default="all",
            choices=["all", "metadata", "metrics"],
            help="Type of evaluation to perform: \n'all': complete evaluation including metadata and performance metrics\n'metadata': model information only (architecture, parameters, etc.)\n'metrics': performance metrics only (accuracy, F1-score, confusion matrix) (default: %(default)s)"
        )
        evaluate_parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cuda", "cpu"],
            help="Computation device:\n'auto': use CUDA if available, else CPU\n'cuda': force GPU\n'cpu': force CPU. (default: auto)"
        )
        evaluate_parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Number of samples processed together in each evaluation step. (default: 8)"
        )
        evaluate_parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="Number of parallel processes for loading data during evaluation. (default: 4)"
        )
        return evaluate_parser

    @staticmethod
    def parse_plot_arguments(subparsers) -> argparse.ArgumentParser:
        """Create parser for plot command with subcommands."""
        plot_parser = subparsers.add_parser("plot", help="Plot utilities")

        plot_subparsers = plot_parser.add_subparsers(dest="plot_command", required=True)
        csv_parser = plot_subparsers.add_parser("csv", help="Plot from CSV file")
        csv_parser.add_argument(
            "--csv-path",
            type=str,
            required=True,
            help="Path to CSV file containing training logs"
        )

        augment_parser = plot_subparsers.add_parser("augments", help="Visualize augmentations")
        augment_parser.add_argument(
            "--config-path",
            type=str,
            help="Path to the transforms config YAML. (default: configs/transforms/default.yaml)"
        )
        augment_parser.add_argument(
            "--image-path",
            type=str,
            required=True,
            help="Path to input image file for augmentation visualization."
        )
        augment_parser.add_argument(
            "--image-size",
            type=int,
            default=224,
            help="Target size for resizing images before applying augmentations. (default: 224)"
        )
        augment_parser.add_argument(
            "--num-samples",
            type=int,
            default=5,
            help="Number of augmented variations to generate from the input image. (default: 5)"
        )
        augment_parser.add_argument(
            "--visual-type",
            type=str,
            default="scroll",
            choices=["scroll", "grid"],
            help="Layout format for displaying augmented images:\n'scroll': scroll arrangement suitable for detailed inspection\n'grid': compact grid layout for quick comparison. (default: %(default)s)"
        )
        return plot_parser

    @staticmethod
    def parse_tools_arguments(subparsers) -> argparse.ArgumentParser:
        """Create parser for tools command with subcommands."""
        tools_parser = subparsers.add_parser("tools", help="Tools and Utilities")

        tools_subparsers = tools_parser.add_subparsers(dest="tool_command", required=True)
        split_parser = tools_subparsers.add_parser("split", help="Split the classes")
        split_parser.add_argument(
            "--input-dir",
            type=str,
            required=True,
            help="Path to input directory containing images to split."
        )
        split_parser.add_argument(
            "--output-dir",
            type=str,
            help="Path to output directory where train/validation splits will be saved. (default: original input directory)" 
        )
        split_parser.add_argument(
            "--split",
            type=float,
            default=0.25,
            help="Fraction of images to allocate to validation set (0.0-1.0). (default: 0.25)"
        )
        split_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Whether to overwrite existing output directory if it already exists. If False and output directory exists, the operation will fail to prevent accidental data loss. Set to True to replace existing splits."
        )

        limit_parser = tools_subparsers.add_parser("limit", help="Limits number of images")
        limit_parser.add_argument(
            "--input-dir",
            type=str,
            required=True,
            help="Path to input directory containing images to limit."
        )
        limit_parser.add_argument(
            "--output-dir",
            type=str,
            help="Path to output directory where limited images will be saved. Used if mode = move. (default: original input directory)"
        )
        limit_parser.add_argument(
            "--max-images",
            type=int,
            default=100,
            help="Maximum number of images to keep per class. (default: 100)"
        )
        limit_parser.add_argument(
            "--mode",
            type=str,
            default="move",
            choices=["move", "delete"],
            help="Action to take with images:\n'move': move images to a separate directory defined in output-dir\n'delete': permanently remove excess images. WARNING: 'delete' mode cannot be undone. (default: move)"
        )
        limit_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Perform a simulation without actually moving or deleting files."
        )

        export_parser = tools_subparsers.add_parser("export", help="Export to different formats")
        export_parser.add_argument(
            "--format",
            type=str,
            default="onnx",
            choices=["onnx", "torchscript"],
            help="Export format for model conversion:\n'onnx': Open Neural Network Exchange format\n'torchscript': PyTorch Mobile TorchScript format optimized for mobile deployment. (default: onnx)"
        )
        export_parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the trained PyTorch model file to export."
        )
        export_parser.add_argument(
            "--output-path",
            type=str,
            help="Path where the exported model will be saved. (default: same path as original model)"
        )
        export_parser.add_argument(
            "--device",
            type=str,
            choices=["auto", "cuda", "cpu"],
            help="Computation device:\n'auto': use CUDA if available, else CPU\n'cuda': force GPU\n'cpu': force CPU. (default: auto)"
        )
        export_parser.add_argument(
            "--opset-version",
            type=int,
            default=11,
            help="ONNX opset version for export compatibility. (default: 11)"
        )

        compress_parser = tools_subparsers.add_parser("compress", help="Compress existing models.")
        compress_parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the trained PyTorch/TorchScript/ONNX model file to compress."
        )
        compress_parser.add_argument(
            "--output-path",
            type=str,
            help="Path where the compressed model will be saved. (default: same path as original model)"
        )
        compress_parser.add_argument(
            "--method",
            type=str,
            default="fp16",
            choices=["quantize", "prune", "fp16"],
            help="Compression method to apply (default: fp16):\nquantize: Reduce model size and improve inference speed by lowering precision (int8).\nprune: Remove less important weights from Linear layers (PyTorch only).\nfp16: Convert model weights to half precision (float16) to reduce memory usage."
        )
        compress_parser.add_argument(
            "--device",
            type=str,
            choices=["auto", "cuda", "cpu"],
            help="Computation device:\n'auto': use CUDA if available, else CPU\n'cuda': force GPU\n'cpu': force CPU. (default: auto)"
        )
        compress_parser.add_argument(
            "--amount",
            type=float,
            default=0.25,
            help="Pruning amount for PyTorch model (default: 0.25)"
        )

        lr_parser = tools_subparsers.add_parser("find-lr", help="Find best LR")
        lr_parser.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="Path to data directory containing the classes."
        )
        lr_parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cuda", "cpu"],
            help="Computation device:\n'auto': use CUDA if available, else CPU\n'cuda': force GPU\n'cpu': force CPU. (default: auto)"
        )
        lr_parser.add_argument(
            "--seed",
            type=int,
            help="Random seed (default: 42)"
        )
        lr_parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="Number of samples processed together in each evaluation step. (default: 16)"
        )
        lr_parser.add_argument(
            "--criterion-type",
            type=str,
            default="crossentropy",
            choices=["crossentropy", "bce"],
            help="Loss function for training:\n'crossentropy': use CrossEntropyLoss for multi-class classification\n 'bce': use BCEWithLogitsLoss for binary classification tasks. Only works with a single class. (default: crossentropy)"
        )
        lr_parser.add_argument(
            "--optimizer-type",
            type=str,
            default="adam",
            choices=["adam", "adamw", "sgd"],
            help="Optimization algorithm:\n'adam': Adam optimizer with adaptive learning rates\n'adamw': Adam with decoupled weight decay\n'sgd': Stochastic Gradient Descent with momentum. (default: adam)"
        )
        lr_parser.add_argument(
            "--min-lr",
            type=float,
            default=0.0000001,
            help="Starting (minimum) learning rate for the range test. (default: 0.0000001)"
        )
        lr_parser.add_argument(
            "--max-lr",
            type=float,
            default=10,
            help="Ending (maximum) learning rate for the range test. (default: 10)"
        )
        lr_parser.add_argument(
            "--num-steps",
            type=int,
            default=50,
            help="Number of steps (iterations) for the learning rate range test. (default: 50)"
        )
        lr_parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="Number of subprocesses for data loading. Higher values can speed up data loading but consume more system memory and CPU. Set to 0 to disable multiprocessing. (default: 4)"
        )
        lr_parser.add_argument(
            "--image-size",
            type=int,
            default=224,
            help="Target size (height and width) in pixels for input images. (default: 224)"
        )
        lr_parser.add_argument(
            "--architecture",
            type=str,
            default="resnet34",
            choices=[
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d',
                'wide_resnet50_2', 'wide_resnet101_2',
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                'densenet121', 'densenet161', 'densenet169', 'densenet201',
                'alexnet',
                'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'
            ],
            help="Neural network architecture for the model backbone (default: resnet34): %(choices)s"
        )
        lr_parser.add_argument(
            "--no-pretrained",
            dest="pretrained",
            action="store_false",
            help="Disable loading of pretrained weights from ImageNet."
        )
        lr_parser.set_defaults(pretrained=True)

        return tools_parser
    
def main():
    """Main entry point for argument parsing."""
    parser = argparse.ArgumentParser(description="Image Classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ArgumentParsers.parse_train_arguments(subparsers)
    ArgumentParsers.parse_retrain_arguments(subparsers)
    ArgumentParsers.parse_inference_arguments(subparsers)
    ArgumentParsers.parse_classify_parser(subparsers)
    ArgumentParsers.parse_evaluation_arguments(subparsers)
    ArgumentParsers.parse_plot_arguments(subparsers)
    ArgumentParsers.parse_tools_arguments(subparsers)

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="show debug messages"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="suppress non-error messages"
    )


    arguments = parser.parse_args()

    if arguments.command == "train" or arguments.command == "retrain":
        Config.config_checks(arguments)

    ArgumentErrors.error_checks(arguments, parser)
 
    return arguments