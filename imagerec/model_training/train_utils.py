import torch
from typing import List, Dict, Tuple, Any
import collections
import random
import math
import os
import csv
import datetime
import logging
logger = logging.getLogger(__name__)

class SubsetDataset(torch.utils.data.Dataset):
    """
    A dataset that wraps an existing dataset and restricts it to a set of indices.

    Args:
        base_dataset (Dataset): Original dataset (e.g., torchvision.datasets.ImageFolder)
        indices (List[int]): Indices to sample from the original dataset
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        indices: List[int],
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices

        # extract original labels from the base dataset
        self.original_labels = [base_dataset[i][1] for i in indices]
        self.targets = self.original_labels

    def __len__(self) -> int:
        """Return the length of indices."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Get a single sample from the subset."""
        image, _ = self.base_dataset[self.indices[idx]]
        label = self.targets[idx]
        return image, label
    

def sample_few_per_class(
    dataset,
    num_of_images_percentage: int,
    class_to_idx: dict,
    include_classes: List[str] = None,
    exclude_classes: List[str] = None
) -> SubsetDataset:
    """
    Sample a subset of images from each class in the dataset.

    Args:
        dataset: Dataset object that returns (sample, label) when indexed.
        num_of_images_percentage (int): Percentage (1-100) of images to sample per class.
        class_to_idx (dict): Mapping from class name to class index.
        include_classes (list, optional): Specific class names to include.
        exclude_classes (list, optional): Specific class names to exclude.

    Returns:
        SubsetDataset: A dataset containing the sampled subset.
    """
    if num_of_images_percentage > 1:
        num_of_images_percentage = num_of_images_percentage / 100.0

    class_to_indices = collections.defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    logger.debug(f"Classes found in dataset: {sorted(class_to_indices.keys())}")
    logger.debug(f"class_to_idx mapping: {class_to_idx}")
    
    if include_classes and exclude_classes:
        logger.error("Use either include_classes or exclude_classes, not both.")
        raise ValueError("Use either include_classes or exclude_classes, not both.")
    elif include_classes:
        target_classes = [class_to_idx[cls] for cls in include_classes if cls in class_to_idx]
        logger.debug(f"Target classes (include): {target_classes}")
        logger.debug(f"Include class names: {include_classes}")
    elif exclude_classes:
        target_classes = [value for key, value in class_to_idx.items() if key not in exclude_classes]
        logger.debug(f"Target classes (exclude): {target_classes}")
        logger.debug(f"Exclude class names: {exclude_classes}")
    else:
        target_classes = list(class_to_idx.values())
        logger.debug(f"Target classes (all): {target_classes}")

    sampled_indices = []
    rng = random.Random()

    for cls in target_classes:
        indices = class_to_indices[cls]
        if len(indices) == 0:
            logger.warning(f"No samples found for class {cls}")
            continue
            
        num_to_sample = max(1, math.ceil(len(indices) * num_of_images_percentage))
        sampled = rng.sample(indices, min(num_to_sample, len(indices)))
        sampled_indices.extend(sampled)
        
        logger.debug(f"Class {cls}: {len(indices)} total, {len(sampled)} sampled")

    logger.info(f"Total sampled indices: {len(sampled_indices)}")
    
    return SubsetDataset(dataset, sampled_indices)

def setup_train_paths(
    base_log_directory: str,
    base_model_directory: str,
    model_name: str = "models",
    add_timestamp: bool = True
) -> tuple[str, str]:
    """
    Create timestamped log and model directories.

    Args:
        base_log_directory (str): Base directory for log files.
        base_model_directory (str): Base directory for saving models.
        model_name (str, optional): Custom model/log identifier. Default is "models".
        add_timestamp (bool): Enable or disable adding timestamp to output folder names.

    Returns:
        tuple:
            - str: Full path to CSV log file.
            - str: Full path to model directory.
    """

    if add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{model_name}_{timestamp}"
    else:
        log_name = model_name
        counter = 1
        path = os.path.join(base_model_directory, log_name)
        while os.path.exists(path):
            log_name = f"{model_name}_{counter}"
            path = os.path.join(base_model_directory, log_name)
            counter += 1

    model_directory = os.path.join(base_model_directory, log_name)
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(base_log_directory, exist_ok=True)

    log_file = os.path.join(base_log_directory, f"{log_name}.csv")

    logger.info(f"Created log file: {log_file}")
    logger.info(f"Created model directory: {model_directory}")

    return log_file, model_directory

def setup_retrain_paths(
    base_log_directory: str,
    model_path: str,
    base_model_directory: str | None = None,
    model_name: str | None = None
) -> tuple[str, str]:
    """
    Create log and model paths for retraining, including automatic backups.

    Args:
        base_log_directory (str): Directory to store log files.
        model_path (str): Path to the existing model file.
        base_model_directory (str | None, optional): Directory for saving retrained models.
            If provided, retrained models will be stored here.
            If None, the retrained model will be stored in the same directory as the original.
        model_name (str | None, optional): Optional custom model name.
            If not provided, the filename (without extension) from model_path is used.

    Returns:
        tuple:
            - str: Path to the log file.
            - str: Path to the retrained model directory.
    """
    if model_path is None:
        logger.error("Model path must be provided")
        raise ValueError("Model path must be provided")
    
    model_name = model_name if model_name else os.path.splitext(os.path.basename(model_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{model_name}_{timestamp}"

    log_file = os.path.join(base_log_directory, f"{log_name}.csv")
    os.makedirs(base_log_directory, exist_ok=True)

    if base_model_directory:
        os.makedirs(base_model_directory, exist_ok=True)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Model directory (retrain): {base_model_directory}")
        return log_file, base_model_directory

    # if base_model_directory not given, create retrained_model directory and place the new model there
    model_directory = os.path.dirname(model_path)

    retrain_directory = os.path.join(model_directory, "retrained_model")
    os.makedirs(retrain_directory, exist_ok=True)

    return log_file, retrain_directory

def model_logs(
    log_file: str,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    validation_loss: float,
    validation_accuracy: float,
    learning_rate : float,
    duration: str
) -> None:
    """
    Append training and validation metrics to a CSV log file.

    Args:
        log_file (str): Path to the CSV log file.
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the epoch.
        train_accuracy (float): Training accuracy for the epoch.
        validation_loss (float): Validation loss for the epoch.
        validation_accuracy (float): Validation accuracy for the epoch.
        learning_rate (float): Current learning rate.
        duration (str): Duration of the epoch in a human-readable format.
    """
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.exists(log_file)

    with open(log_file, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy", "Learning Rate", "Duration"])
        writer.writerow([epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, learning_rate, duration])


def save_model(
        model: torch.nn.Module,
        path: str,
        architecture: str,
        image_size: int,
        pretrained: bool,
        class_to_idx: Dict[str, int],
        optimizer: torch.optim.Optimizer = None,
        epoch: int = None
) -> None:
    """
    Save a PyTorch model along with its metadata.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): Destination file path where the checkpoint will be saved.
        architecture (str): Name of the model architecture.
        image_size (int): Input image size used for training.
        pretrained (bool): Whether the model was initialized with pretrained weights.
        class_to_idx (Dict[str, int]): Mapping of class names to indices.
        optimizer (Optional[torch.optim.Optimizer], optional): Optimizer to save state for. Defaults to None.
        epoch (Optional[int], optional): Epoch number to store in checkpoint. Defaults to None.

    Returns:
        None
    """
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save = {
        "model_state_dict" : model.state_dict(),
        "class_to_idx" : class_to_idx,
        "num_classes" : len(class_to_idx),
        "architecture" : architecture,
        "pretrained" : pretrained,
        "image_size" : image_size
    }

    if optimizer:
        save["optimizer_state_dict"] = optimizer.state_dict()
        logger.debug("Optimizer state added to checkpoint.")
    if epoch is not None:
        save["epoch"] = epoch
        logger.debug(f"Epoch {epoch} added to checkpoint.")


    torch.save(save, path)
    logger.info(f"Model Saved: {path}")