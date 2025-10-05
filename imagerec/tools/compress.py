import os
import torch
import onnx
import logging
logger = logging.getLogger(__name__)

from imagerec.common import utils
from imagerec.common import builder

def compress_model(
    model_path: str,
    output_path: str = None,
    method: str = "fp16",
    device: str = "auto",
    amount: float = 0.3,
):
    """
    Compress a model (Torch, TorchScript, or ONNX).

    Args:
        model_path (str): Path to the model file.
        output_path (str): Path where the compressed model will be saved.
        method (str): Compression method ("quantize", "prune", "fp16").
        amount (float, optional): Pruning amount (0.0-1.0).
    """

    extension = os.path.splitext(model_path)[-1].lower()
    base = os.path.splitext(os.path.basename(model_path))[0]
    compress_name = base + "_compressed" + extension

    if not output_path:
        output_path = os.path.join(os.path.dirname(model_path), compress_name)
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, compress_name)

    device, use_cuda = utils.set_device(device=device)
    logger.info(f"Compressing {extension} model at {model_path} with method={method}")

    if extension == ".pt":
        if utils.is_torchscript_model(model_path): # TorchScript
            logger.info("Loading TorchScript model...")
            model = torch.jit.load(model_path, map_location=device)

            if method == "quantize":
                model.eval()
                try:
                    import torchao
                    model = torchao.quantization.quantize_(
                        model,
                        torchao.quantization.int8_dynamic_activation_int8_weight()
                    )

                    logger.info("Used torchao quantization")

                except Exception:
                    logger.warning("torchao not available, using legacy torch.ao.quantization (deprecated)")
                    # if any(isinstance(m, torch.nn.Conv2d) for m in model.modules()):
                    #     raise RuntimeError("Legacy torch.ao.quantization.quantize_dynamic cannot safely quantize models with Conv2d layers. Please install TorchAO and use Int8WeightOnlyConfig.")
                    model = torch.ao.quantization.quantize_dynamic(
                        model,
                        qconfig_spec={torch.nn.Linear, torch.nn.Conv2d},
                        dtype=torch.qint8
                    )

            elif method == "prune":
                logger.warning("Pruning is not directly supported for TorchScript models")
                raise ValueError("Pruning is not directly supported for TorchScript models")

            elif method == "fp16":
                model = model.half()
                logger.info("Successfully compressed model.")
            
            else:
                logger.error(f"Unsupported method '{method}' for Torch. Supported: quantize, prune, fp16")
                raise ValueError(f"Unsupported method '{method}' for Torch. Supported: quantize, prune, fp16")
            
            model.save(output_path)

            logger.info(f"Compressed TorchScript model saved to {output_path}")

        else: # PyTorch
            logger.info("Loading PyTorch model...")
            model_values = builder.load_model(
                model_path=model_path,
                device=device
            )
            metadata = {key: value for key, value in model_values.items() if key not in ['model']}
            model = model_values["model"]  # must be a torch.nn.Module
            metadata["num_classes"] = len(model_values["class_to_idx"].keys())
            logger.info("PyTorch model successfully loaded.")

            if method == "quantize":
                model.eval()
                try:
                    from torchao.quantization import quantize_, Int8WeightOnlyConfig
                    quantize_(
                        model,
                        Int8WeightOnlyConfig()
                    )

                    logger.info("Used torchao quantization")

                except:
                    logger.warning("torchao not available, using legacy torch.ao.quantization (deprecated)")
                    if any(isinstance(m, torch.nn.Conv2d) for m in model.modules()):
                        raise RuntimeError("Legacy torch.ao.quantization.quantize_dynamic cannot safely quantize models with Conv2d layers. Please install TorchAO and use Int8WeightOnlyConfig.")
                    model = torch.ao.quantization.quantize_dynamic(
                        model,
                        qconfig_spec={torch.nn.Linear, torch.nn.Conv2d},
                        dtype=torch.qint8
                    )
                logger.info("Successfully compressed model.")
            elif method == "prune":
                if not 0 <= amount <= 1.0:
                    logger.error(f"The amount specified should be in the range 0.0 - 1.0: {amount}")
                    raise ValueError(f"The amount specified should be in the range 0.0 - 1.0: {amount}")
                
                import torch.nn.utils.prune as prune

                parameters_to_prune = []
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                        parameters_to_prune.append((module, "weight"))
                
                if parameters_to_prune:
                    prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=amount,
                    )
                    for module, parameter_name in parameters_to_prune:
                        prune.remove(module, parameter_name)
                    
                else:
                    logger.warning("No pruning parameters found.")

            elif method == "fp16":
                model = model.half()
                logger.info("Successfully compressed model.")
            
            else:
                logger.error(f"Unsupported method '{method}' for Torch. Supported: quantize, prune, fp16")
                raise ValueError(f"Unsupported method '{method}' for Torch. Supported: quantize, prune, fp16")

            
            save = {"model_state_dict" : model.state_dict(), **metadata}
            torch.save(save, output_path)

            logger.info(f"Compressed Torch model saved to {output_path}")

    elif extension == ".onnx":
        if method == "quantize":
            logger.error("Quantization is not directly supported for ONNX models. Only 'fp16' is supported.")
            raise ValueError("Quantization is not directly supported for ONNX models. Only 'fp16' is supported.")
        
        elif method == "prune":
            logger.error("Pruning is not directly supported for ONNX models. Only 'fp16' is supported.")
            raise ValueError("Pruning is not directly supported for ONNX models. Only 'fp16' is supported.")

        elif method == "fp16":
            try:
                from onnxconverter_common import float16
                model = onnx.load(model_path)
                model_fp16 = float16.convert_float_to_float16(
                    model, 
                    keep_io_types=True,
                    disable_shape_infer=False
                )
                onnx.save(model_fp16, output_path)
                logger.info("Successfully compressed model.")
            except ImportError:
                logger.error("onnxconverter-common is required for ONNX fp16 conversion. Install with: pip install onnxconverter-common")
                raise ImportError("onnxconverter-common is required for ONNX fp16 conversion. Install with: pip install onnxconverter-common")
        else:
            logger.error(f"Unsupported method '{method}' for ONNX. Supported: quantize, fp16")
            raise ValueError(f"Unsupported method '{method}' for ONNX. Supported: quantize, fp16")

        logger.info(f"Compressed ONNX model saved to {output_path}")

    else:
        logger.error(f"Unknown format '{extension}'. The only supported formats are torch, torchscript and onnx created from this program.")
        raise ValueError(f"Unknown format '{extension}'. The only supported formats are torch, torchscript and onnx created from this program.")

    try:
        original_size = os.path.getsize(model_path)
        compressed_size = os.path.getsize(output_path)
        compression_ratio = compressed_size / original_size

        logger.info(f"Compression completed. Original: {original_size/1024/1024:.2f}MB, Compressed: {compressed_size/1024/1024:.2f}MB, Ratio: {compression_ratio:.3f}")
    except:
        logger.warning(f"Could not calculate compression ratio.")