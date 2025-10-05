import shutil
import random
import pathlib
import logging
logger = logging.getLogger(__name__)

def split_dataset_by_class(
    input_dir: str,
    output_dir: str,
    split: float = 0.2,
    overwrite: bool = False
):
    """
    Split a dataset into training and validation subsets by class.
    Each class folder inside the input directory will be divided into training and validation sets based on the provided split ratio. 
    The resulting directory structure will look like:

        output_dir/
            ├── train/
            │   ├── class1/
            │   ├── class2/
            │   ...
            └── val/
                ├── class1/
                ├── class2/
                ...

    Args:
        input_dir (str): Path to the dataset directory containing class subfolders.
        output_dir (str, optional): Path to save the split dataset. Defaults to the same as input_dir if not provided.
        split (float): Fraction of images to use for validation (Ex. 0.2 = 20% validation).
        overwrite (bool): If True, existing train/val directories will be removed before splitting.

    """
    
    input_dir = pathlib.Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
        logger.warning("No output directory provided. Using input directory for output.")
    else:
        output_dir = pathlib.Path(output_dir)
    
    train_directory = pathlib.Path(output_dir) / "train"
    validation_directory = pathlib.Path(output_dir) / "val"

    if overwrite:
        shutil.rmtree(train_directory, ignore_errors=True)
        shutil.rmtree(validation_directory, ignore_errors=True)
        logger.info("Removed existing train/val directories as argument 'overwrite' is True.")

    # if overwrite is false and directories exist
    if train_directory.exists() or validation_directory.exists():
        logger.error("Train/Validation directories already exist. Aborting to avoid overwrite.")
        raise FileExistsError("Train/Validation directories already exist. Aborting to avoid overwrite.")

    train_directory.mkdir(parents=True, exist_ok=True)
    validation_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directories: {train_directory}, {validation_directory}")

    for class_directory in pathlib.Path(input_dir).iterdir():
        if not class_directory.is_dir():
            continue

        class_name = class_directory.name

        images = list(
            list(class_directory.glob("*.jpg")) +
            list(class_directory.glob("*.jpeg")) +
            list(class_directory.glob("*.png"))
        )

        if not images:
            logger.warning(f"No images found in {class_name}. Skipping.")
            continue

        random.shuffle(images)

        split_images = int(len(images) * split)
        
        train_images = images[split_images:]
        validation_images = images[:split_images]

        train_class_directory = train_directory / class_name
        validation_class_directory = validation_directory / class_name

        train_class_directory.mkdir(parents=True, exist_ok=True)
        validation_class_directory.mkdir(parents=True, exist_ok=True)

        for image in train_images:
            shutil.copy2(image, train_class_directory / image.name)
        for image in validation_images:
            shutil.copy2(image, validation_class_directory / image.name)

        logger.info(f"Train -> {class_name} : {len(train_images)}")
        logger.info(f"Validation -> {class_name} : {len(validation_images)}")
        logger.info("Dataset successfully split into train and validation sets.")