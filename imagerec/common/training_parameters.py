import torch
import logging
logger = logging.getLogger(__name__)

def set_criterion(criterion_type: str = "crossentropy"):
    """
    Sets the loss function (criterion).

    Args:
        criterion_type (str): Type of criterion to use.
            Options:
                - "crossentropy": Multi-class classification.
                - "bce": Binary classification.

    Returns:
        torch.nn.Module: Initialized loss function.
    """

    if criterion_type == "crossentropy":
        return torch.nn.CrossEntropyLoss()
    elif criterion_type == "bce":
        return torch.nn.BCEWithLogitsLoss()
    else:
        logger.error(f"Unknown criterion type: {criterion_type}")
        raise ValueError(f"Unknown criterion type: {criterion_type}")

def set_scheduler(
    optimizer: torch.optim,
    scheduler_type: str = 'step',
    epochs: int = 50
):
    """
    Sets the scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to apply the scheduler to.
        scheduler_type (str): Type of scheduler. Options:
            - "none": No scheduler.
            - "step": StepLR, decreases LR every 15 epochs by gamma=0.1.
            - "cosine": CosineAnnealingLR over total epochs.
            - "plateau": ReduceLROnPlateau based on validation loss.
        epochs (int): Number of training epochs (used in cosine schedule).

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: Scheduler instance.
    """
    if scheduler_type == "none":
        return None
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    else:
        logger.error(f"Unknown scheduler type: {scheduler_type}")
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def set_optimizer(
    model: torch.nn.Module,
    lr: float = 0.001,
    optimizer_type: str = "adam",
    strategy="full"
):
    """
    Sets the optimizer and its respective strategy.

    Args:
        model (torch.nn.Module): Model whose parameters need optimization.
        lr (float): Learning rate for the optimizer.
        optimizer_type (str): Optimizer type. Options:
            - "adam"
            - "adamw"
            - "sgd"
        strategy (str): Training strategy for parameter updates. Options:
            - "full": Train all model parameters.
            - "freeze_backbone": Freeze backbone, train only classifier.
            - "differential_lr": Lower LR for backbone, higher LR for classifier.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    classifier_keywords = ['fc', 'classifier', 'head', 'linear']

    if strategy == "full":
        parameters = model.parameters()

    elif strategy == "freeze_backbone":
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # unfreeze classifier layers (last layers)
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in classifier_keywords):
                param.requires_grad = True
        
        parameters = filter(lambda p: p.requires_grad, model.parameters())

    elif strategy == "differential_lr":
        backbone_parameters, classifier_parameters = [], []
        for name, parameter in model.named_parameters():
            if any(x in name.lower() for x in classifier_keywords):
                classifier_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        parameters = [
            {'params': backbone_parameters, 'lr': lr * 0.01},
            {'params': classifier_parameters, 'lr': lr}
        ]
    else:
        logger.error(f"Unknown strategy type: {strategy}")
        raise ValueError(f"Unknown strategy type: {strategy}")

    if optimizer_type == "adam":
        return torch.optim.Adam(parameters)  # no global lr
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(parameters)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(parameters, momentum=0.9)
    else:
        logger.error(f"Unknown optimizer type: {optimizer_type}")
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")