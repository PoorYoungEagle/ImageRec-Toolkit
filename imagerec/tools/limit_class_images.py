import shutil
import pathlib
import random
import logging
logger = logging.getLogger(__name__)

def limit_class_images(
    input_dir: str,
    output_dir: str = None,
    max_images: int = 100,
    mode: str = "move",
    dry_run: bool = False
):
    """
    Limit the number of images per class by deleting or moving the excess.
    It keeps a maximum number of images for each class folder inside the input path.
    If a class contains more images than the allowed maximum, the extra images will either be deleted or moved to a backup/output directory, depending on the mode selected.

    Args:
        input_dir (str): Path to the input dataset directory (containing class subfolders).
        output_dir (str, optional): Path to the output directory for moved images. If None and mode is "move", defaults to "<input_dir>_backup".
        max_images (int): Maximum number of images to keep per class.
        mode (str): Action to perform on excess images:
                    - "delete": Permanently remove extra images.
                    - "move": Move extra images to the output directory.
        dry_run (bool): If True, only log the actions without performing them.
        
    Supported File Types:
        Only '.jpg', '.jpeg', and '.png' files are processed.
    """
    
    input_path = pathlib.Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input folder not found: {input_dir}")
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    if mode == "move" and output_dir is None:
        output_dir = input_dir.rstrip("/\\") + "_backup"
        logger.info(f"No output folder provided. Using backup folder: {output_dir}")

    if output_dir and not dry_run:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for class_directory in input_path.iterdir():
        if not class_directory.is_dir():
            continue

        images = list(list(class_directory.glob("*.jpg")) + list(class_directory.glob("*.jpeg")) + list(class_directory.glob("*.png")))
        total_images = len(images)

        if total_images <= max_images:
            logger.info(f"[SKIP] {class_directory.name} has {total_images} images (<= {max_images})")
            continue

        random.shuffle(images)
        keep = images[:max_images]
        discard = images[max_images:]

        if dry_run:
            for image in discard:
                if mode == "move":
                    logger.info(f"[DRY-RUN] Would move: {image}")
                elif mode == "delete":
                    logger.info(f"[DRY-RUN] Would delete: {image}")
            continue

        for image in discard:
            if mode == "delete":
                image.unlink()
                logger.debug(f"Deleted: {image}")
            elif mode == "move":
                destination_directory = output_path / class_directory.name
                destination_directory.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image), str(destination_directory / image.name))
                logger.debug(f"Moved: {image} -> {destination_directory}")

    logger.info("Successfully limited the number of images per class.")