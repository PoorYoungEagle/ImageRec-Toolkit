import shutil
import os
import pathlib
import logging
logger = logging.getLogger(__name__)

from imagerec.inference import predict
from imagerec.common import builder
from imagerec.common import utils
from imagerec.inference import infer_utils

def classify(
    model_path: str,
    input_dir: str,
    output_dir: str = None,
    config_labels: str = None,
    device: str = "auto",
    confidence_threshold: float = 0.5,
    delete_original: bool = False
):
    """
    Classify images in a folder using a PyTorch (.pt) or ONNX (.onnx) model.

    Args:
        model_path (str): Path to the model file (.pt or .onnx).
        input_dir (str): Path to folder of images.
        output_dir (str, optional): Path to store classified results. Defaults to "classified_results" inside the input path if not given.
        config_labels (str, optional): Path to config file with labels (for ONNX only).
        device (str, optional): Computation device ("auto", "cpu", or "cuda").
        confidence_threshold (float): Minimum confidence/probability required for a prediction to be considered valid (0.0 - 1.0). Will create a directory called 'Uncertain' if image below confidence threshold.
        delete_original (bool, optional): Whether to delete input images after classification.

    """
    extension = os.path.splitext(model_path)[-1].lower()
    device, use_cuda = utils.set_device(device=device)
    if not output_dir:
        output_dir = pathlib.Path(input_dir) / "classified_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"No output path provided. Using default: {output_dir}")
    else:
        logger.info(f"Output directory: {output_dir}")

    if extension == ".pt":
        if utils.is_torchscript_model(model_path):
            logger.info("Loading TorchScript model...")
            model_values = builder.load_model_torchscript(
                model_path=model_path,
                config_labels=config_labels,
                device=device
            )
            logger.info("TorchScript model successfully loaded.")
        else:
            logger.info(f"Loading PyTorch model from {model_path}")
            model_values = builder.load_model(
                model_path=model_path,
                device=device
            )
            logger.info("PyTorch model successfully loaded.")

        model = model_values["model"]
        model.eval()

        class_to_idx = model_values["class_to_idx"]
        classifier = predict.TorchClassifier(
            model=model,
            class_to_idx=class_to_idx,
            device=device,
            image_size=model_values["image_size"],
            top_n=1
        )
        logger.info("PyTorch model successfully loaded.")

    elif extension == ".onnx":
        logger.info(f"Loading ONNX model from {model_path}")
        model_values = builder.load_model_onnx(
            model_path=model_path,
            config_labels=config_labels,
            use_cuda=use_cuda
        )

        class_to_idx = model_values["class_to_idx"]
        classifier = predict.ONNXClassifier(
            session=model_values["session"],
            input_name=model_values["input_name"],
            output_name=model_values["output_name"],
            class_to_idx=class_to_idx,
            image_size=model_values["image_size"],
            top_n=1
        )
        logger.info("ONNX model and config labels successfully loaded.")
    else:
        logger.error(f"Invalid extension format")
        raise ValueError(f"Invalid extension format")

    for name in list(class_to_idx.keys()):
        name_directory = pathlib.Path(output_dir) / str(name)
        name_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory for class '{name}': {name_directory}")

    results = classifier.predict_folder_images(
        folder_path=input_dir
    )
    logger.info(f"Classification completed. {len(results)} images processed.")

    infer_utils.infer_directory_images(results=results)
    for result in results:
        predicted_class = result["top_prediction"]
        top_probability = result["top_probability"]

        if top_probability < confidence_threshold:
            predicted_class = "Uncertain"

        destination_directory = pathlib.Path(output_dir) / predicted_class
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination_directory = pathlib.Path(output_dir) / predicted_class
        try:
            shutil.copy2(result["image_path"], destination_directory)
            logger.debug(f"Copied {result['image_path']} : {destination_directory}")
        except Exception as e:
            logger.error(f"Failed to copy {result['image_path']}: {e}")

        if delete_original:
            try:
                pathlib.Path(result["image_path"]).unlink(missing_ok=True)
                logger.debug(f"Deleted original image: {result['image_path']}")
            except Exception as e:
                logger.error(f"Failed to delete {result['image_path']}: {e}")