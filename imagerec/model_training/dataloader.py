import os
import torchvision
import torch
from typing import List, Tuple
import logging
logger = logging.getLogger(__name__)

from imagerec.common import albumentation_transforms
from imagerec.model_training import train_utils

def get_dataloader(
        data_directory: str,
        transforms_config: str = "imagerec/configs/transforms/default.yaml",
        batch_size: int = 32,
        image_size: int = 224,
        num_workers: int = 4,
        num_of_images_percentage: int = 100,
        use_augments: bool = True,
        pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Creates and returns PyTorch DataLoaders for training and validation datasets.

    Args:
        data_directory (str): Path to data directory containing train/ and validation/ folders.
        transforms_config (str): Path to the transforms yaml file. 
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        image_size (int, optional): Target image size for resizing/cropping. Defaults to 224.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        use_augments (bool, optional): Whether to apply data augmentation to training set. Defaults to True.
        pin_memory (bool, optional): Whether to use CUDA pinned memory for loading. Defaults to True.
        num_of_images_percentage (int, optional): Percentage of images per class to use for training. Defaults to 100.
        seed (int, optional): Random seed. Defaults to 42

    Returns:
          Tuple[DataLoader, DataLoader, int]:
            - train_loader: DataLoader for the training dataset.
            - validation_loader: DataLoader for the validation dataset.
            - num_classes: Number of distinct classes in the dataset.
    """

    train_directory = os.path.join(data_directory, "train")
    if not os.path.exists(train_directory):
        logger.error(f"Training directory not found: {train_directory}")
        raise FileNotFoundError(f"Training directory not found: {train_directory}")

    if os.path.isdir(os.path.join(data_directory, "validation")):
        validation_directory = os.path.join(data_directory, "validation")
    elif os.path.isdir(os.path.join(data_directory, "val")):
        validation_directory = os.path.join(data_directory, "val")
    else:
        logger.error(f"Validation directory not found: {validation_directory}")
        raise FileNotFoundError(f"Validation directory not found: {validation_directory}")
    
    logger.debug(f"Training directory: {train_directory}")
    logger.debug(f"Validation directory: {validation_directory}")
    
    train_transform = albumentation_transforms.TrainTransform(
        image_size,
        config_path=transforms_config,
        is_training=True,
        use_augments=use_augments
    )
    validation_transform = albumentation_transforms.TrainTransform(
        image_size,
        is_training=False
    )

    logger.debug(f"Train transform: {train_transform}")
    logger.debug(f"Validation transform: {validation_transform}")

    train_dataset = torchvision.datasets.ImageFolder(
        train_directory,
        transform=train_transform
    )
    if num_of_images_percentage < 100:
        logger.info(f"Sampling {num_of_images_percentage}% of images per class for training.")
        train_dataset = train_utils.sample_few_per_class(
            dataset=train_dataset,
            num_of_images_percentage=num_of_images_percentage,
            class_to_idx=train_dataset.class_to_idx,
            include_classes=list(train_dataset.class_to_idx.keys())
        )

    validation_dataset = torchvision.datasets.ImageFolder(
        validation_directory,
        transform=validation_transform
    )

    validation_dataset.class_to_idx = train_dataset.class_to_idx

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.debug(f"Class-to-idx mapping: {train_dataset.class_to_idx}")

    logger.info(f"Number of Objects:  {len(train_dataset.classes)}")
    logger.info(f"Number of Training Samples:  {len(train_dataset)}")
    logger.info(f"Number of Validation Samples:  {len(validation_dataset)}")

    return train_loader, validation_loader, len(train_dataset.classes)

def get_retraining_dataloader(
    data_directory: str,
    merged_class_to_idx: dict,
    new_classes: List[str],
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_augments: bool = True,
    pin_memory: bool = True,
    skip_missing_classes: bool = False,
    num_of_images_percentage: int = 50
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Creates PyTorch DataLoaders for retraining with old classes downsampled
    and new classes fully included.

    Args:
        data_directory (str): Path to data directory containing train/ and validation/ folders.
        merged_class_to_idx (dict): Old and new classes mapped.
        new_classes (List[str]): List of new class names to include fully.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        image_size (int, optional): Target image size for resizing/cropping. Defaults to 224.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        use_augments (bool, optional): Whether to apply augmentations to training images. Defaults to True.
        pin_memory (bool, optional): Whether to use CUDA pinned memory for loading. Defaults to True.
        num_of_images_percentage (int, optional): Percentage of images to sample for old classes. Defaults to 50.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader]:
            - train_loader: DataLoader for retraining (old classes + new classes).
            - validation_loader: DataLoader for retraining validation set.
    """

    train_directory = os.path.join(data_directory, "train")
    if not os.path.exists(train_directory):
        logger.error(f"Training directory not found: {train_directory}")
        raise FileNotFoundError(f"Training directory not found: {train_directory}")

    if os.path.isdir(os.path.join(data_directory, "validation")):
        validation_directory = os.path.join(data_directory, "validation")
    elif os.path.isdir(os.path.join(data_directory, "val")):
        validation_directory = os.path.join(data_directory, "val")
    else:
        logger.error(f"Validation directory not found: {validation_directory}")
        raise FileNotFoundError(f"Validation directory not found: {validation_directory}")
    
    logger.debug(f"Training directory: {train_directory}")
    logger.debug(f"Validation directory: {validation_directory}")
    
    train_transform = albumentation_transforms.TrainTransform(
        image_size,
        is_training=True,
        use_augments=use_augments
    )
    validation_transform = albumentation_transforms.TrainTransform(
        image_size,
        is_training=False
    )

    logger.debug(f"Train transform: {train_transform}")
    logger.debug(f"Validation transform: {validation_transform}")

    train_dataset = torchvision.datasets.ImageFolder(
        train_directory,
        transform=train_transform
    )
    validation_dataset = torchvision.datasets.ImageFolder(
        validation_directory,
        transform=validation_transform
    )

    all_classes = set(train_dataset.classes + validation_dataset.classes)
    missing_classes = [cls for cls in all_classes if cls not in merged_class_to_idx]
    
    def remove_unwanted_classes(dataset, allowed_classes):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        dataset.samples = [(path, merged_class_to_idx[idx_to_class[target]]) for path, target in dataset.samples if idx_to_class[target] in allowed_classes]
        dataset.targets = [s[1] for s in dataset.samples]
        dataset.classes = list(allowed_classes)
        dataset.class_to_idx = {cls: merged_class_to_idx[cls] for cls in allowed_classes}
        return dataset

    if missing_classes:
        if skip_missing_classes:
            logger.warning(f"Skipping classes not in merged_class_to_idx: {missing_classes}")
            allowed_classes = set(merged_class_to_idx.keys())
            train_dataset = remove_unwanted_classes(train_dataset, allowed_classes)
            validation_dataset = remove_unwanted_classes(validation_dataset, allowed_classes)
        else:
            logger.error(f"Classes in data but not in merged_class_to_idx: {missing_classes}")
            raise ValueError(f"Classes in data but not in merged_class_to_idx: {missing_classes}")

    
    logger.debug(f"Initial train classes: {train_dataset.classes}")
    logger.debug(f"Initial validation classes: {validation_dataset.classes}")
    logger.debug(f"Merged class_to_idx mapping: {merged_class_to_idx}")

    train_dataset.class_to_idx = merged_class_to_idx.copy()
    train_dataset.classes = list(merged_class_to_idx.keys())
    train_dataset.samples = [(path, merged_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in train_dataset.samples]
    train_dataset.targets = [label for _, label in train_dataset.samples]

    validation_dataset.class_to_idx = merged_class_to_idx.copy()
    validation_dataset.classes = list(merged_class_to_idx.keys())
    validation_dataset.samples = [(path, merged_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in validation_dataset.samples]
    validation_dataset.targets = [label for _, label in validation_dataset.samples]

    class_to_idx = train_dataset.class_to_idx

    for cls in new_classes:
        if cls not in class_to_idx:
            logger.error(f"Class '{cls}' not found in the dataset")
            raise ValueError(f"Class '{cls}' not found in the dataset")
    
    old_classes = [cls for cls in class_to_idx if cls not in new_classes]
    new_class_indices = [class_to_idx[cls] for cls in new_classes]

    logger.debug(f"Old classes: {old_classes}")
    logger.debug(f"New classes: {new_classes}")

    old_subset = train_utils.sample_few_per_class(
        dataset=train_dataset,
        num_of_images_percentage=num_of_images_percentage,
        class_to_idx=class_to_idx,
        include_classes=old_classes
    )

    new_indices = [i for i, (_, label) in enumerate(train_dataset) if label in new_class_indices]
    new_subset = train_utils.SubsetDataset(train_dataset, new_indices)

    validation_old_indices = [i for i, (_, label) in enumerate(validation_dataset) if label in [class_to_idx[c] for c in old_classes]]
    validation_new_indices = [i for i, (_, label) in enumerate(validation_dataset) if label in new_class_indices]
    
    validation_old = train_utils.SubsetDataset(validation_dataset, validation_old_indices)
    validation_new = train_utils.SubsetDataset(validation_dataset, validation_new_indices)
    
    concatenated_train_dataset = torch.utils.data.ConcatDataset([old_subset, new_subset])
    concatenated_validation_dataset = torch.utils.data.ConcatDataset([validation_old, validation_new])

    all_train_targets = [label for _, label in concatenated_train_dataset]
    all_validation_targets = [label for _, label in concatenated_validation_dataset]

    train_loader = torch.utils.data.DataLoader(
        concatenated_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    validation_loader = torch.utils.data.DataLoader(
        concatenated_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.debug(f"Concatenated Train Dataset: {all_train_targets}")
    logger.debug(f"Concatenated Validation Dataset: {all_validation_targets}")
    logger.debug(f"Original class mapping: {dict(sorted(train_dataset.class_to_idx.items()))}")
    logger.debug(f"Merged class mapping: {dict(sorted(merged_class_to_idx.items()))}")
    logger.debug(f"Total classes after merge: {len(class_to_idx)}")
    logger.debug(f"Old classes ({len(old_classes)}): {old_classes}")
    logger.debug(f"New classes ({len(new_classes)}): {new_classes}")

    logger.info(f"Training samples -> Old: {len(old_subset)} ({num_of_images_percentage}% of original), New: {len(new_subset)} (100% of original)")
    logger.info(f"Validation samples -> Old: {len(validation_old)}, New: {len(validation_new)}")

    return train_loader, validation_loader