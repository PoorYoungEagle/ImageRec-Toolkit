import os
import logging
logger = logging.getLogger(__name__)

from imagerec.inference import predict
from imagerec.inference import infer_utils
from imagerec.common import builder
from imagerec.common import utils

def infer_class(
    model_path: str,
    input_path: str,
    device: str = "auto",
    top_n: int = 3,
    include_classes: list = None,
    exclude_classes: list = None,
    config_labels: str = None
):
    """
    Run inference using a trained model (PyTorch '.pt', ONNX '.onnx', TorchScript) on a single image or a folder of images.

    Args:
        model_path (str): Path to the model file (.pt or .onnx).
        input_path (str): Path to image file or folder of images.
        device (str): Device to run inference on (e.g. "cpu", "cuda").
        top_n (int): Number of top predictions to return.
        include_classes (list, optional): Restrict predictions to these classes.
        exclude_classes (list, optional): Exclude these classes from predictions.
        config_labels (str, optional): Path to ONNX labels config.
    """

    logger.info("Starting inference:")

    extension = os.path.splitext(model_path)[-1].lower()
    device, use_cuda = utils.set_device(device)

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
            logger.info("Loading PyTorch model...")
            model_values = builder.load_model(
                model_path=model_path,
                device=device
            )
            logger.info("PyTorch model successfully loaded.")

        model = model_values["model"]
        model.eval()

        classifier = predict.TorchClassifier(
            model=model,
            class_to_idx=model_values["class_to_idx"],
            device=device,
            image_size=model_values["image_size"],
            top_n=top_n,
            include_classes=include_classes,
            exclude_classes=exclude_classes
        )

    elif extension == ".onnx":
        logger.info("Loading ONNX model...")
        model_values = builder.load_model_onnx(
            model_path=model_path,
            config_labels=config_labels,
            use_cuda=use_cuda
        )

        classifier = predict.ONNXClassifier(
            session=model_values["session"],
            input_name=model_values["input_name"],
            output_name=model_values["output_name"],
            class_to_idx=model_values["class_to_idx"],
            image_size=model_values["image_size"],
            top_n=top_n,
            include_classes=include_classes,
            exclude_classes=exclude_classes
        )
        logger.info("ONNX model successfully loaded.")
    else:
        logger.error(f"Invalid extension format: {extension}")
        raise ValueError(f"Invalid extension format: {extension}")

    if os.path.isfile(input_path):
        logger.info(f"Running inference on single image: {input_path}")
        result = classifier.predict_image(image_path=input_path)
        infer_utils.infer_image(result)

    elif os.path.isdir(input_path):
        logger.info(f"Running inference on folder of images: {input_path}")
        results = classifier.predict_folder_images(folder_path=input_path)
        infer_utils.infer_directory_images(results)
    
    else:
        logger.error(f"Invalid input path: {input_path}")
        raise ValueError(f"Invalid input path: {input_path}")