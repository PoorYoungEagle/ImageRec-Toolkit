import torch
import os
import pathlib
import numpy as np
import logging
logger = logging.getLogger(__name__)

from imagerec.common import albumentation_transforms

class ObjectClassifier:
    """
    Base class for models.

    Args:
        class_to_idx (dict): Mapping from class names to indices.
        top_n (int, optional): Number of top predictions to return. Default: 3.
        include_classes (list[str], optional): If provided, restrict predictions
            to this subset of class names.
        exclude_classes (list[str], optional): If provided, exclude predictions
            for these class names.
    """

    def __init__(
        self,
        class_to_idx,
        top_n: int = 3,
        include_classes = None,
        exclude_classes = None
    ):
        self.class_to_idx = class_to_idx
        self.top_n = top_n
        self.include_classes = include_classes
        self.exclude_classes = exclude_classes

    def _filter_predictions(self, probabilities: torch.Tensor):
        """
        Apply class filtering (include/exclude) and return the top-N predictions.

        Args:
            probabilities (torch.Tensor): Class probability tensor (shape: [num_classes]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Top-N probabilities (shape: [top_n_actual])  
                - Corresponding class indices (shape: [top_n_actual])  

        """
        num_classes = len(self.class_to_idx)
        valid_classes_mask = torch.ones(num_classes, dtype=torch.bool)

        if self.include_classes and self.exclude_classes:
            logger.error("Use either include_classes or exclude_classes, not both.")
            raise ValueError("Use either include_classes or exclude_classes, not both.")

        elif self.include_classes:
            include_indices = []
            for cls in self.include_classes:
                if cls not in self.class_to_idx:
                    logger.warning(f"Included class '{cls}' not found in class_to_idx.")
                else:
                    include_indices.append(self.class_to_idx[cls])
            valid_classes_mask = torch.zeros(num_classes, dtype=torch.bool)
            valid_classes_mask[include_indices] = True

        elif self.exclude_classes:
            exclude_indices = []
            for cls in self.exclude_classes:
                if cls not in self.class_to_idx:
                    logger.warning(f"Excluded class '{cls}' not found in class_to_idx.")
                else:
                    exclude_indices.append(self.class_to_idx[cls])
            valid_classes_mask[exclude_indices] = False

        filtered_probabilities = probabilities * valid_classes_mask.to(probabilities.device)

        # taking top n predictions
        top_n_actual = min(self.top_n, valid_classes_mask.sum().item())
        top_probabilities, top_indices = torch.topk(filtered_probabilities, k=top_n_actual)

        logger.debug(f"Filtered predictions: {top_n_actual} classes selected ")
        logger.debug(f"Total Classes: {num_classes}")
        
        return top_probabilities, top_indices
    
    def _get_image_files(
        self,
        folder_path: str,
    ):
        """
        Collect all valid image files in a given folder.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            List[pathlib.Path]: List of image file paths.
        """
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        formats = [".jpg", ".jpeg", ".png"]

        image_files = set()

        for format in formats:
            image_files.update(pathlib.Path(folder_path).glob(f'*{format}'))
            image_files.update(pathlib.Path(folder_path).glob(f'*{format.upper()}'))

        image_files = list(image_files)

        if not image_files:
            logger.error(f"There are no image files found in {folder_path}")
            raise ValueError(f"There are no image files found in {folder_path}")
        
        logger.info(f"Processing {len(image_files)} images")

        return image_files
    
    def predict_image(self, image_path: str):
        """Must be implemented by subclasses"""
        raise NotImplementedError
    
    def predict_folder_images(
        self,
        folder_path: str,
    ):
        """
        Predict all images in a folder.

        Args:
            folder_path (str): Path to the folder.

        Returns:
            List[dict]: List of prediction results for each image.
        """
        image_files = self._get_image_files(folder_path=folder_path)

        results = []
        for image_file in image_files:
            try:
                result = self.predict_image(str(image_file))
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(results)}/{len(image_files)} images successfully.")

        return results


class TorchClassifier(ObjectClassifier):
    """
    Initialize the TorchClassifier.

    Args:
        model (torch.nn.Module): Trained PyTorch model for inference.
        class_to_idx (dict): Mapping of class names to indices.
        device (torch.device): Device to run inference on (CPU or GPU).
        image_size (int, optional): Input image size for preprocessing. Defaults to 224.
        top_n (int, optional): Number of top predictions to return. Defaults to 3.
        include_classes (List[str], optional): Classes to include for filtering. Defaults to None.
        exclude_classes (List[str], optional): Classes to exclude for filtering. Defaults to None.
    """
    def __init__(
        self,
        model,
        class_to_idx,
        device: torch.device,
        image_size: int = 224,
        top_n: int = 3,
        include_classes = None, 
        exclude_classes = None
    ) -> dict:

        super().__init__(class_to_idx, top_n, include_classes, exclude_classes)
        self.model = model
        self.device = device
        self.image_size = image_size

    def predict_image(
            self,
            image_path: str,
    ) -> dict:
        """
        Predict the class of a single image using the trained PyTorch model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: Prediction result containing:
                - "image_path": str, path of the image
                - "top_prediction": str, predicted top class
                - "top_confidence": str, formatted confidence percentage
                - "all_predictions": List[dict], ranked predictions with fields:
                    - "rank": int
                    - "class": str
                    - "probability": float
                    - "confidence": str
        """
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.debug(f"Running inference on image: {image_path}")
        
        transform = albumentation_transforms.InferenceTransform(image_size=self.image_size)
        image_tensor = transform(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        logger.debug(f"Model output shape: {outputs.shape}")
        logger.debug(f"Probabilities tensor shape: {probabilities.shape}")
        
        top_probabilities, top_indices = self._filter_predictions(probabilities)
        top_probabilities = top_probabilities.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()

        idx_to_class = {value: key for key, value in self.class_to_idx.items()}

        predictions = []

        for i, (probability, index) in enumerate(zip(top_probabilities, top_indices), 1):
            predictions.append({
                "rank" : i,
                "class" : idx_to_class[index],
                "probability" : float(probability),
                "confidence" : f"{probability * 100:.2f}%"
            })

        result = {
            "image_path" : image_path,
            "top_prediction" : predictions[0]["class"],
            "top_probability" : predictions[0]["probability"],
            "top_confidence" : predictions[0]["confidence"],
            "all_predictions" : predictions
        }

        logger.info(f"Predicted '{result['top_prediction']}' with confidence {result['top_confidence']} for image: {image_path}")

        return result
    
class ONNXClassifier(ObjectClassifier):
    """
    Initialize the ONNXClassifier.

    Args:
        session (onnxruntime.InferenceSession): ONNX Runtime session object.
        input_name (str): Name of the model input tensor.
        output_name (str): Name of the model output tensor.
        class_to_idx (Dict[str, int]): Mapping of class names to indices.
        image_size (int, optional): Input image size for preprocessing. Defaults to 224.
        top_n (int, optional): Number of top predictions to return. Defaults to 3.
        include_classes (List[str], optional): Classes to include for filtering. Defaults to None.
        exclude_classes (List[str], optional): Classes to exclude for filtering. Defaults to None.
    """
    def __init__(
        self,
        session,
        input_name,
        output_name,
        class_to_idx,
        image_size: int = 224,
        top_n: int = 3,
        include_classes = None,
        exclude_classes = None
    ):
        super().__init__(class_to_idx, top_n, include_classes, exclude_classes)
        self.session = session
        self.image_size = image_size
        self.input_name = input_name
        self.output_name = output_name

    def predict_image(
            self,
            image_path: str
    )-> dict:
        """
        Predict the class of a single image using the trained ONNX model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: Prediction result containing:
                - "image_path": str, path of the image
                - "top_prediction": str, predicted top class
                - "top_confidence": str, formatted confidence percentage
                - "all_predictions": List[dict], ranked predictions with fields:
                    - "rank": int
                    - "class": str
                    - "probability": float
                    - "confidence": str
        """
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.debug(f"Running inference on image: {image_path}")
        
        transform = albumentation_transforms.InferenceTransform(image_size=self.image_size)
        image_tensor = transform(image_path)
        image_np = image_tensor.cpu().numpy()

        if len(image_np.shape) == 3:
            image_np = np.expand_dims(image_np, axis=0)
            logger.debug("Added batch dimension to input tensor.")

        ort_inputs = {self.input_name: image_np}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        outputs = ort_outs[0]

        logger.debug("ONNX model inference completed.")

        outputs_torch = torch.from_numpy(outputs)
        probabilities = torch.nn.functional.softmax(outputs_torch, dim=1)

        top_probabilities, top_indices = self._filter_predictions(probabilities)
        top_probabilities = top_probabilities.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()

        idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        predictions = []
        for i, (probability, index) in enumerate(zip(top_probabilities, top_indices), 1):
            predictions.append({
                "rank": i,
                "class": idx_to_class[index],
                "probability": float(probability),
                "confidence": f"{probability * 100:.2f}%"
            })

        result = {
            "image_path": image_path,
            "top_prediction": predictions[0]["class"],
            "top_probability" : predictions[0]["probability"],
            "top_confidence": predictions[0]["confidence"],
            "all_predictions": predictions
        }

        logger.info(f"Predicted '{result['top_prediction']}' with confidence {result['top_confidence']} for image: {image_path}")

        return result