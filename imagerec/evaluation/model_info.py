import os
import torch
import logging
logger = logging.getLogger(__name__)

from imagerec.common import builder


def load_metadata(
    model_path: str,
    device: torch.device
):
    """
    Load model metadata from a saved model.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to map the model onto ("cpu" or "cuda").
    
    Returns:
        dict: Metadata dictionary containing model details such as:
            - architecture (str): Model architecture name.
            - pretrained (bool): Whether the model was pretrained.
            - image_size (int): Input image size expected by the model.
            - class_to_idx (dict): Mapping of class names to indices.
            - model (torch.nn.Module): The loaded model.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model metadata from: {model_path}")
    metadata = builder.load_model(
        model_path=model_path,
        device=device
    )

    logger.info("Model metadata successfully loaded")
    return metadata

def print_metadata(metadata: dict):
    """
    Prints metadata.

    Args:
        metadata (dict): Metadata dictionary as returned by 'load_metadata'.
    """
    logger.info("Model Metadata: ")
    logger.info(f"Number of Classes: {len(metadata['class_to_idx'])}")
    logger.info(f"Architecture: {metadata['architecture']}")
    logger.info(f"Pretrained: {metadata['pretrained']}")
    logger.info(f"Input Image Size: {metadata['image_size']}")

    logger.info("Class to Index Mapping:")
    for cls, idx in metadata["class_to_idx"].items():
        logger.info(f"{idx} : {cls}")