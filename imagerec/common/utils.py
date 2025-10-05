import torch
import torchvision
import random
import numpy as np
import os
import datetime
import shutil
import csv
import json
import logging
logger = logging.getLogger(__name__)

from imagerec.common import albumentation_transforms

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed across libraries.

    Args:
        seed (int, optional): The seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(device: str = "auto") -> tuple[torch.device, bool]:
    """
    Configure the computation device.

    Args:
        device (str, optional): Choice of device. Default is "auto".
            - "auto": Select CUDA if available, otherwise CPU.
            - "cuda": Force CUDA if available, fallback to CPU if not.
            - "cpu": Force CPU.

    Returns:
        tuple:
            - torch.device: Selected device object.
            - bool: Whether CUDA is available/selected.
    """
    if device == "auto":
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Using CPU.")
            return torch.device("cpu"), False
        logger.info("Using CUDA (auto-detected).")
        return torch.device("cuda"), True
    elif device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU.")
            return torch.device("cpu"), False
        logger.info("Using CUDA (forced).")
        return torch.device("cuda"), True
    else:
        logger.info("Using CPU.")
        return torch.device("cpu"), False

def is_torchscript_model(model_path: str) -> bool:
    """
    Checks whether the .pt model is torch or torchscript.

    Args:
        model_path(str): Path to the model.

    Returns:
        bool: True for torchscript, False for torch.
    """
    try:
        torch.jit.load(model_path)
        return True
    except RuntimeError:
        return False

def save_config(
    config: dict,
    config_name: str,
    path: str
) -> None:
    """
    Save configuration dictionary as a JSON file.

    Args:
        config (dict): Configuration dictionary to save.
        config_name (str): Name of the config
        path (str): Path to the JSON file.

    """

    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, f"{config_name}.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Configuration saved to {path}")

def dataloader(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_cuda: bool = False
) -> tuple[torch.utils.data.DataLoader, int]:
    """
    Create a PyTorch DataLoader with albumentation transforms applied.

    Args:
        data_dir (str): Path to dataset directory.
        batch_size (int, optional): Number of samples per batch. Default: 32.
        image_size (int, optional): Target image size for transformations. Default: 224.
        num_workers (int, optional): Number of worker for data loading. Default: 4.
        use_cuda (bool, optional): Use pinned memory for GPU training. Default: False.

    Returns:
        tuple:
            - DataLoader: PyTorch DataLoader for the dataset.
            - int: Number of classes in the dataset.
    """
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    transform = albumentation_transforms.TrainTransform(
        image_size,
        is_training=False
    )
    dataset = torchvision.datasets.ImageFolder(
        data_dir,
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )

    logger.info(f"Number of classes: {len(dataset.classes)}")
    logger.info(f"Number of samples: {len(dataset)}")

    return loader, len(dataset.classes)