import os
import torch
import logging
logger = logging.getLogger(__name__)

def infer_image(result):
    """
    Log the inference results for a single image.

    Args:
        result (dict): Dictionary containing inference results:
            - "image_path" (str): Path to the image.
            - "top_prediction" (str): Class label of the top prediction.
            - "top_confidence" (float): Confidence score of the top prediction.
            - "all_predictions" (list of dict): Each dict contains:
                - "rank" (int): Prediction rank.
                - "class" (str): Predicted class label.
                - "confidence" (float): Confidence score for the prediction.
    """

    image_path = result["image_path"]
    top_prediction = result["top_prediction"]
    top_confidence = result["top_confidence"]
    all_predictions = result["all_predictions"]

    image_name = os.path.basename(image_path)
    logger.info(f"Image name: {image_name}")

    for prediction in all_predictions:
        logger.info(f"Rank: {prediction['rank']}")
        logger.info(f"Class: {prediction['class']}")
        logger.info(f"Confidence: {prediction['confidence']}")

def infer_directory_images(results):
    """
    Log the inference results for a directory of images.

    Args:
        results (list of dict): List of inference results where each element is structured as in 'infer_image'.
    """
    if not results:
        logger.warning("There are no results present")
        return
    
    logger.info(f"Processing {len(results)} results from directory inference.")
    for result in results:
        infer_image(result)
        logger.info("\n")