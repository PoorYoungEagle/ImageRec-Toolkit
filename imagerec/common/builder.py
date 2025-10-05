import torchvision
import torch
import os
import json
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

# mapping name to model and preloaded weights
MODELS = {
    # ResNet Family
    "resnet18" : (torchvision.models.resnet18, torchvision.models.ResNet18_Weights),
    "resnet34" : (torchvision.models.resnet34, torchvision.models.ResNet34_Weights),
    "resnet50" : (torchvision.models.resnet50, torchvision.models.ResNet50_Weights),
    "resnet101" : (torchvision.models.resnet101, torchvision.models.ResNet101_Weights),
    "resnet152" : (torchvision.models.resnet152, torchvision.models.ResNet152_Weights),

    # ResNeXt
    "resnext50_32x4d" : (torchvision.models.resnext50_32x4d, torchvision.models.ResNeXt50_32X4D_Weights),
    "resnext101_32x8d" : (torchvision.models.resnext101_32x8d, torchvision.models.ResNeXt101_32X8D_Weights),
    "resnext101_64x4d" : (torchvision.models.resnext101_64x4d, torchvision.models.ResNeXt101_64X4D_Weights),

    # Wide ResNet
    "wide_resnet50_2" : (torchvision.models.wide_resnet50_2, torchvision.models.Wide_ResNet50_2_Weights),
    "wide_resnet101_2" : (torchvision.models.wide_resnet101_2, torchvision.models.Wide_ResNet101_2_Weights),

    # EfficientNet
    "efficientnet_b0" : (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights),
    "efficientnet_b1" : (torchvision.models.efficientnet_b1, torchvision.models.EfficientNet_B1_Weights),
    "efficientnet_b2" : (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights),
    "efficientnet_b3" : (torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights),
    "efficientnet_b4" : (torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights),
    "efficientnet_b5" : (torchvision.models.efficientnet_b5, torchvision.models.EfficientNet_B5_Weights),
    "efficientnet_b6" : (torchvision.models.efficientnet_b6, torchvision.models.EfficientNet_B6_Weights),
    "efficientnet_b7" : (torchvision.models.efficientnet_b7, torchvision.models.EfficientNet_B7_Weights),

    # MobileNet
    "mobilenet_v2" : (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights),
    "mobilenet_v3_small" : (torchvision.models.mobilenet_v3_small, torchvision.models.MobileNet_V3_Small_Weights),
    "mobilenet_v3_large" : (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights),

    # VGG
    "vgg11" : (torchvision.models.vgg11, torchvision.models.VGG11_Weights),
    "vgg11_bn" : (torchvision.models.vgg11_bn, torchvision.models.VGG11_BN_Weights),
    "vgg13" : (torchvision.models.vgg13, torchvision.models.VGG13_Weights),
    "vgg13_bn" : (torchvision.models.vgg13_bn, torchvision.models.VGG13_BN_Weights),
    "vgg16" : (torchvision.models.vgg16, torchvision.models.VGG16_Weights),
    "vgg16_bn" : (torchvision.models.vgg16_bn, torchvision.models.VGG16_BN_Weights),
    "vgg19" : (torchvision.models.vgg19, torchvision.models.VGG19_Weights),
    "vgg19_bn" : (torchvision.models.vgg19_bn, torchvision.models.VGG19_BN_Weights),

    # DenseNet
    "densenet121" : (torchvision.models.densenet121, torchvision.models.DenseNet121_Weights),
    "densenet161" : (torchvision.models.densenet161, torchvision.models.DenseNet161_Weights),
    "densenet169" : (torchvision.models.densenet169, torchvision.models.DenseNet169_Weights),
    "densenet201" : (torchvision.models.densenet201, torchvision.models.DenseNet201_Weights),

    # AlexNet
    "alexnet" : (torchvision.models.alexnet, torchvision.models.AlexNet_Weights),

    # ShuffleNet
    "shufflenet_v2_x0_5" : (torchvision.models.shufflenet_v2_x0_5, torchvision.models.ShuffleNet_V2_X0_5_Weights),
    "shufflenet_v2_x1_0" : (torchvision.models.shufflenet_v2_x1_0, torchvision.models.ShuffleNet_V2_X1_0_Weights),
    "shufflenet_v2_x1_5" : (torchvision.models.shufflenet_v2_x1_5, torchvision.models.ShuffleNet_V2_X1_5_Weights),
    "shufflenet_v2_x2_0" : (torchvision.models.shufflenet_v2_x2_0, torchvision.models.ShuffleNet_V2_X2_0_Weights),

    # MNASNet
    "mnasnet0_5" : (torchvision.models.mnasnet0_5, torchvision.models.MNASNet0_5_Weights),
    "mnasnet0_75" : (torchvision.models.mnasnet0_75, torchvision.models.MNASNet0_75_Weights),
    "mnasnet1_0" : (torchvision.models.mnasnet1_0, torchvision.models.MNASNet1_0_Weights),
    "mnasnet1_3" : (torchvision.models.mnasnet1_3, torchvision.models.MNASNet1_3_Weights),
}


def build_model(
    num_of_classes: int,
    architecture: str = "resnet18",
    pretrained: bool = True
) -> torch.nn.Module:
    """
    Build and return a torchvision classification model.

    Args:
        num_of_classes (int): Number of output classes for the final layer.
        architecture (str): Model architecture to use. Default is "resnet18".
        pretrained (bool): If True, load pretrained ImageNet weights. Default is True.

    Returns:
        torch.nn.Module: Model with modified final layer for the given number of classes.
    """
    if architecture not in MODELS:
        logger.error(f"Unknown architecture {architecture}\nChoose from {list(MODELS)}")
        raise ValueError(f"Unknown architecture {architecture}\nChoose from {list(MODELS)}")
    
    logger.info(f"Building model: {architecture} (pretrained={pretrained}, num_classes={num_of_classes})")

    model, weights = MODELS[architecture]

    weights = weights.DEFAULT if pretrained else None
    model = model(weights=weights)

    if hasattr(model, "fc"): # for resnet and efficientnet
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_of_classes)
        logger.debug(f"Replaced model.fc with Linear(in_features={in_features}, out_features={num_of_classes})")

    elif hasattr(model, "classifier"): # for mobile or vgg
        # .classifier can be sequential or linear

        if isinstance(model.classifier, torch.nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, num_of_classes)
            logger.debug(f"Replaced last layer of model.classifier Sequential with Linear(in_features={in_features}, out_features={num_of_classes})")
        else:
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_of_classes)
            logger.debug(f"Replaced model.classifier with Linear(in_features={in_features}, out_features={num_of_classes})")
    else:
        logger.error("Architecture does not contain a factory or classifier attribute, please add the logic in builder.py")
        raise RuntimeError("Architecture does not contain a factory or classifier attribute, please add the logic in builder.py")
    
    return model

def load_model(
        model_path: str,
        device: torch.device
) -> Dict[str, Any]:
    """
    Load a trained PyTorch model along with its metadata.

    Args:
        model_path (str): Path to the saved model (.pth or .pt).
        device (torch.device): The device to map the model to.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "model": The loaded model (torch.nn.Module)
            - "class_to_idx": Mapping of class labels to indices
            - "image_size": Image size used during training
            - "pretrained": Whether pretrained weights were used
            - "architecture": Model architecture name
    """
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_state_dict = checkpoint["model_state_dict"]
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]
    pretrained = checkpoint["pretrained"]
    architecture = checkpoint["architecture"]
    image_size = checkpoint["image_size"]

    if model_state_dict is None or num_classes is None or architecture is None:
        logger.error("PyTorch model missing metadata. Please select the model which has been trained using this program.")
        raise ValueError("PyTorch model missing metadata. Please select the model which has been trained using this program.")

    model = build_model(
        num_of_classes = num_classes,
        architecture = architecture,
        pretrained = pretrained
    )

    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)

    logger.info("Model loaded successfully.")
    logger.info(f"Architecture: {architecture}")
    logger.info(f"Pretrained: {pretrained}")
    logger.info(f"Classes: {list(class_to_idx.keys())}")

    model_values = {
        "model" : model,
        "class_to_idx" : class_to_idx,
        "image_size" : image_size,
        "pretrained" : pretrained,
        "architecture" : architecture,
    }

    return model_values

def load_model_for_retraining(
        model_path: str,
        device: torch.device,
        new_classes: list,
        data_directory: str
) -> Dict[str, Any]:
    """
    Load a trained model checkpoint and expand its classifier to handle
    new classes for retraining.

    Args:
        model_path (str): Path to the saved checkpoint (.pth or .pt).
        device (torch.device): The device to map the model to.
        new_classes (List[str]): List of new class names to add. If empty, existing classes from old model will be used.
        data_directory (str): Path to dataset root (expects "train/" subfolder).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "model": Retrained model
            - "class_to_idx": Merged class-to-index mapping
            - "new_classes": Classes newly added compared to old model
            - "image_size": Input image size
            - "pretrained": Whether pretrained weights were used
            - "architecture": Model architecture name
    """
    model_values = load_model(
        model_path=model_path,
        device=device
    )
    old_class_to_idx = model_values["class_to_idx"]
    model = model_values["model"]
    
    train_directory = os.path.join(data_directory, "train")
    temp_dataset = torchvision.datasets.ImageFolder(train_directory)
    new_data_class_to_idx = temp_dataset.class_to_idx
    
    logger.info(f"All classes in directory: {list(new_data_class_to_idx.keys())}")
    
    old_classes = set(old_class_to_idx.keys())
    
    if new_classes:
        added_new_classes = set(new_classes)
    else:
        # detecting new classes
        all_classes = [d for d in os.listdir(train_directory) if os.path.isdir(os.path.join(train_directory, d))]
        added_new_classes = set(all_classes) - set(old_class_to_idx.keys())

        if not added_new_classes:
            logger.warning("No new classes detected. Using existing classes only.")
            added_new_classes = set(old_class_to_idx.keys())


    new_classes = list(added_new_classes - old_classes)

    logger.info(f"Number of old classes: {len(old_classes)}")
    logger.info(f"New classes to add: {len(new_classes)} - {new_classes}")

    allowed_classes = old_classes.union(added_new_classes)
    merged_class_to_idx = {key: value for key, value in new_data_class_to_idx.items() if key in allowed_classes}
    merged_class_to_idx = {cls: i for i, cls in enumerate(sorted(list(merged_class_to_idx.keys())))}
    
    logger.debug(f"Merged class_to_idx mapping: {merged_class_to_idx}")
    logger.info(f"Expanded to {len(merged_class_to_idx)} classes.")

    if hasattr(model, "fc"):
        old_layer = model.fc
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, torch.nn.Linear):
            old_layer = model.classifier
        elif isinstance(model.classifier, torch.nn.Sequential):
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], torch.nn.Linear):
                    old_layer = model.classifier[i]
                    layer_attr = ('classifier', i)
                    break
            else:
                logger.error("No Linear layer found in model.classifier Sequential.")
                raise ValueError("No Linear layer found in model.classifier Sequential.")
    else:
        logger.error("Model does not have a recognizable output layer.")
        raise ValueError("Model does not have a recognizable output layer.")
    
    num_features = old_layer.in_features
    new_layer = torch.nn.Linear(num_features, len(merged_class_to_idx))
    torch.nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')
    torch.nn.init.zeros_(new_layer.bias)

    with torch.no_grad():
        for class_name, old_idx in old_class_to_idx.items():
            if class_name in merged_class_to_idx:
                new_idx = merged_class_to_idx[class_name]
                logger.debug(f"Mapping {class_name}: old_idx {old_idx} -> new_idx {new_idx}")
                new_layer.weight[new_idx] = old_layer.weight[old_idx]
                new_layer.bias[new_idx] = old_layer.bias[old_idx]

    if hasattr(model, "fc"):
        model.fc = new_layer
    elif hasattr(model, "classifier"):
        model.classifier = new_layer
    else:
        classifier, idx = layer_attr
        model.classifier[idx] = new_layer
    
    logger.info("Model architecture has been expanded successfully")
    model = model.to(device)

    retrained_model_values = {
        "model" : model,
        "class_to_idx" : merged_class_to_idx,
        "new_classes" : new_classes,
        "image_size" : model_values["image_size"],
        "pretrained" : model_values["pretrained"],
        "architecture" : model_values["architecture"],
    }
    
    return retrained_model_values

def load_model_torchscript(
    model_path: str,
    config_labels: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load a TorchScript model and its configuration.

    Args:
        model_path (str): Path to the TorchScript model file (.pt same as PyTorch).
        config_labels (str): Path to the JSON config file containing 'class_to_idx' and 'image_size'.
        device (torch.device): The device to map the model to.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "model": The loaded model (torch.jit.ScriptModule)
            - "class_to_idx": Class-to-index mapping
            - "image_size": Required image size
    """
    if not os.path.exists(model_path):
        logger.error(f"TorchScript model not found: {model_path}")
        raise FileNotFoundError(f"TorchScript model not found: {model_path}")
    if not config_labels:
        logger.error(f"Please include the Config labels (.json file)")
        raise FileNotFoundError(f"Please include the Config labels (.json file)")
    if not os.path.exists(config_labels) or os.path.splitext(config_labels)[-1].lower() != ".json":
        logger.error(f"Config labels not found, it must be included for torchscript: {config_labels}")
        raise FileNotFoundError(f"Config labels not found, it must be included for torchscript: {config_labels}")

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    with open(config_labels, "r") as f:
        data = json.load(f)

    if "class_to_idx" not in data or "image_size" not in data:
        logger.error(f"Config labels file {config_labels} must contain 'class_to_idx' and 'image_size' keys.")
        raise KeyError(f"Config labels file {config_labels} must contain 'class_to_idx' and 'image_size' keys.")

    return {
        "model": model,
        "class_to_idx": data["class_to_idx"],
        "image_size": data["image_size"]
    }

def load_model_onnx(
    model_path: str,
    config_labels: str,
    use_cuda: bool
) -> Dict[str, Any]:
    """
    Load an ONNX model and its configuration.

    Args:
        model_path (str): Path to the ONNX model file (.onnx).
        config_labels (str): Path to the JSON config file containing 'class_to_idx' and 'image_size'.
        use_cuda (bool): Whether to use CUDA if available.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "session": ONNXRuntime inference session
            - "input_name": Model input tensor name
            - "output_name": Model output tensor name
            - "class_to_idx": Class-to-index mapping
            - "image_size": Required image size
    """

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime is not installed. Please install it with 'pip install onnxruntime'.")
        raise ImportError("onnxruntime is not installed. Please install it with 'pip install onnxruntime'.")
    
    if not os.path.exists(model_path):
        logger.error(f"ONNX model not found: {model_path}")
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not config_labels:
        logger.error(f"Please include the Config labels (.json file)")
        raise FileNotFoundError(f"Please include the Config labels (.json file)")
    if not os.path.exists(config_labels) or os.path.splitext(config_labels)[-1].lower() != ".json":
        logger.error(f"Config labels not found, it must be included for onnx: {config_labels}")
        raise FileNotFoundError(f"Config labels not found, it must be included for onnx: {config_labels}")
    
    providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

    logger.info(f"Loading ONNX model from: {model_path}")

    session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    with open(config_labels, 'r') as file:
        data = json.load(file)

    if "class_to_idx" not in data or "image_size" not in data:
        logger.error(f"Config labels file {config_labels} must contain 'class_to_idx' and 'image_size' keys.")
        raise KeyError(f"Config labels file {config_labels} must contain 'class_to_idx' and 'image_size' keys.")

    model_values = {
        "session": session,
        "input_name": input_name,
        "output_name": output_name,
        "class_to_idx": data["class_to_idx"],
        "image_size": data["image_size"]
    }

    logger.info("ONNX model loaded successfully")

    return model_values