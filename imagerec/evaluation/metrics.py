from sklearn import metrics
import numpy as np
import tqdm
import torch
import logging
logger = logging.getLogger(__name__)

def compute_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list,
    device: torch.device
):
    """
    Compute and log evaluation metrics for a trained model.

    Args:
        model (torch.nn.Module): Trained PyTorch model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for evaluation dataset.
        class_names (list of str): List of class labels corresponding to dataset indices.
        device (torch.device): Device to run evaluation on ("cpu" or "cuda").
    """
    model.eval()
    all_predictions = []
    all_labels = []

    logger.info("Starting evaluation loop")
    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    confusion = metrics.confusion_matrix(all_labels, all_predictions)
    report = metrics.classification_report(all_labels, all_predictions, target_names=class_names)

    logger.info("Evaluation Complete")
    logger.info(f"Accuracy Score: {accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{confusion}")
    logger.info(f"Classification Report:\n{report}")

    np.fill_diagonal(confusion, 0)  # ignore correct predictions

    confusions = []
    for i, row in enumerate(confusion):
        for j, value in enumerate(row):
            if i != j and value > 0:
                confusions.append((i, j, value))

    confusions.sort(key=lambda x: x[2], reverse=True)

    if confusions:
        logger.info("Top Confusions:")
        for idx, (true_idx, pred_idx, count) in enumerate(confusions, 1):
            if idx > 50:
                logger.debug(f"{idx:2d}. {class_names[true_idx]} -> {class_names[pred_idx]} : {count}")
                continue
            logger.info(f"{idx:2d}. {class_names[true_idx]} -> {class_names[pred_idx]} : {count}")
    else:
        logger.info("No significant confusions found (perfect classification).")