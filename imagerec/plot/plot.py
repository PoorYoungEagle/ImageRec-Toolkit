import logging
logger = logging.getLogger(__name__)

from imagerec.plot import plot_augments
from imagerec.plot import plot_csv

def plot_values(
    plot_command: str,
    csv_path: str = None,
    image_path: str = None,
    config_path: str = None,
    visual_type: str = "scroll",
    num_samples: int = 5,
    image_size: int = 224
):
    """
    Display plots or visualizations.

    This function supports two main plot commands:
        1. csv - Generates plots from a training log CSV file.
           - Plots loss & accuracy, learning rate, and training duration.
        2. augments - Visualizes image augmentations.
           - Supports two visualization modes:
               - "scroll": Display augmentations in a scrolling sequence.
               - "grid": Display augmentations in a grid layout.

    Args:
        plot_command (str): Type of plot to display ("csv" or "augments").
        csv_path (str, optional): Path to the CSV file (required if plot_command="csv").
        image_path (str, optional): Path to input image (required if plot_command="augments")
        config_path (str, optional): Path to augmentation configuration file.
        visual_type (str, optional): Visualization style for augments ("scroll" or "grid").
        num_samples (int, optional): Number of augmented samples to generate.
        image_size (int, optional): Size to which images are resized for visualization.
    """

    if plot_command == "csv":
        if csv_path is None:
            logger.error("csv_path argument must be given.")
            raise ValueError("csv_path argument must be given.")

        logger.info(f"Generating plots from CSV file: {csv_path}")
        
        plot_csv.plot_loss_accuracy(csv_path)
        plot_csv.plot_learning_rate(csv_path)
        plot_csv.plot_duration(csv_path)

    elif plot_command == "augments":
        if image_path is None:
            logger.error("image_path argument must be given.")
            raise ValueError("image_path argument must be given.")
        
        logger.info(f"Visualizing augmentations in {visual_type} mode from {image_path}")
        if visual_type == "scroll":
            plot_augments.visualize_scroll(
                image_path=image_path,
                image_size=image_size,
                config_path=config_path,
                num_samples=num_samples
            )
        elif visual_type == "grid":
            plot_augments.visualize_grid(
                image_path=image_path,
                image_size=image_size,
                config_path=config_path,
                num_samples=num_samples
            )
        else:
            logger.error(f"Unknown visual type: {visual_type}")
            raise ValueError(f"Unknown visual type: {visual_type}")

    else:
        logger.error(f"Unknown plot command: {plot_command}")
        raise ValueError(f"Unknown plot command: {plot_command}")
