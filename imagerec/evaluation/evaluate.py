import logging
logger = logging.getLogger(__name__)

from imagerec.evaluation import metrics
from imagerec.evaluation import model_info
from imagerec.common import utils

def evaluate_model(
    model_path: str,
    data_dir: str = None,
    evaluate_type: str = "all",
    device: str = "auto",
    batch_size: int = 16,
    num_workers: int = 4
):
    """
    Execute evaluation for a trained model.

    Depending on the 'arguments.type', this function:
      - Prepares a dataloader for evaluation.
      - Prints model metadata.
      - Computes evaluation metrics such as accuracy, precision, recall, etc.

    Args:
        model_path (str): Path to the saved model file.
        data_dir (str): Directory containing evaluation dataset.
        evaluate_type (str): Execution type ("all", "metadata", "metrics").
        device (str): Target device ("cpu", "cuda", or "auto").
        batch_size (int): Number of samples per batch for dataloader.
        num_workers (int): Number of worker processes for dataloader.
    """

    device, use_cuda = utils.set_device(
        device=device
    )
    metadata = model_info.load_metadata(
        model_path=model_path,
        device=device
    )
    logger.info(f"Model metadata loaded from {model_path}")

    if data_dir:
        dataloader, _ = utils.dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=metadata["image_size"],
            num_workers=num_workers,
            use_cuda=use_cuda
        )
    logger.info("Dataloader loaded successfully.")

    if evaluate_type == "all":
        logger.info("Printing metadata and computing metrics")
        model_info.print_metadata(
            metadata=metadata
        )
        if data_dir:
            metrics.compute_metrics(
                model=metadata["model"],
                loader=dataloader,
                class_names=list(metadata["class_to_idx"].keys()),
                device=device
            )
        else:
            logger.warning("No data_dir argument given, computing metrics of model cannot be done.")
            
    elif evaluate_type == "metadata":
        logger.info("Printing metadata only")
        model_info.print_metadata(
            metadata=metadata
        )
    elif evaluate_type == "metrics":
        if data_dir:
            logger.info("Computing metrics only")
            metrics.compute_metrics(
                model=metadata["model"],
                loader=dataloader,
                class_names=list(metadata["class_to_idx"].keys()),
                device=device
            )
        else:
            logger.error("No data_dir argument given, computing metrics of model cannot be done.")
            raise ValueError("No data_dir argument given, computing metrics of model cannot be done.")
    else:
        logger.error(f"Type is not recognizable {evaluate_type}")
        raise ValueError(f"Type is not recognizable {evaluate_type}")