import torch
import os
import onnx
import logging
logger = logging.getLogger(__name__)

from imagerec.common import builder
from imagerec.common import utils

def export_to_onnx(
    model_path: str,
    output_path: str = None,
    device: str = "auto",
    opset_version: int = 11
) -> str:
    """
    Export a trained PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Output path for the exported ONNX file.
        device (str): Computation device ("cpu", "cuda", "auto").
        opset_version (int): ONNX opset version.

    Returns:
        str: Path to the exported ONNX file.
    """
    if not output_path:
        model_dir = os.path.dirname(model_path)

        onnx_dir = os.path.join(model_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(model_path))[0] + ".onnx"
        output_path = os.path.join(onnx_dir, base_name)

    elif os.path.isdir(output_path):
        base_name = os.path.splitext(os.path.basename(model_path))[0] + ".onnx"
        output_path = os.path.join(output_path, base_name)

    device, _ = utils.set_device(device=device)
    model_values = builder.load_model(
        model_path=model_path,
        device=device
    )
    model = model_values["model"]
    model.eval()
    class_to_idx = model_values["class_to_idx"]
    image_size = model_values["image_size"]

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=opset_version
        )
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise

    config = {
        "image_size" : image_size,
        "class_to_idx" : class_to_idx
    }
    utils.save_config(
        config=config,
        config_name=os.path.splitext(os.path.basename(model_path))[0] + "_onnx_labels",
        path=os.path.dirname(output_path)
    )

    logger.info(f"Successfully exported to ONNX model: {output_path}")
    return output_path

def export_to_torchscript(
    model_path: str,
    output_path: str = None,
    device: str = "auto"
) -> str:
    """
    Export a trained PyTorch model to TorchScript format for PyTorch Mobile.

    Args:
        model_path (str): Path to the trained PyTorch model checkpoint.
        output_path (str): Output path for the exported TorchScript file.
        device (str): Computation device ("cpu", "cuda", "auto").

    Returns:
        str: Path to the exported TorchScript (.pt) file.
    """

    if not output_path:
        model_dir = os.path.dirname(model_path)

        torchscript_dir = os.path.join(model_dir, "torchscript")
        os.makedirs(torchscript_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(model_path))[0] + "_torchscript.pt"
        output_path = os.path.join(torchscript_dir, base_name)

    elif os.path.isdir(output_path):
        base_name = os.path.splitext(os.path.basename(model_path))[0] + "_torchscript.pt"
        output_path = os.path.join(output_path, base_name)

    device, _ = utils.set_device(device=device)
    
    model_values = builder.load_model(
        model_path=model_path,
        device=device
    )
    model = model_values["model"]
    model.eval()
    model.to(device)
    class_to_idx = model_values["class_to_idx"]
    image_size = model_values["image_size"]

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)

    config = {
        "image_size" : image_size,
        "class_to_idx" : class_to_idx
    }
    utils.save_config(
        config=config,
        config_name=os.path.splitext(os.path.basename(model_path))[0] + "_torchscript_labels",
        path=os.path.dirname(output_path)
    )

    logger.info(f"Successfully exported TorchScript model: {output_path}")
    
    return output_path