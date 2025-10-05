import torch
import numpy as np
import tqdm
import logging
logger = logging.getLogger(__name__)

from imagerec.common import training_parameters
from imagerec.common import utils
from imagerec.common import builder
from imagerec.plot import plot_utils

def find_lr(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    min_lr=1e-7,
    max_lr=10,
    num_steps=100
):
    """
    Finds a suitable learning rate range by gradually increasing the LR and tracking loss. Method is based from the paper 'Cyclical Learning Rates for Training Neural Networks' (2017).

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader providing the training data.
        optimizer: Optimizer to update model parameters.
        criterion: Loss function.
        device: Computation device ('cpu' or 'cuda').
        min_lr (float): Starting learning rate.
        max_lr (float): Maximum learning rate.
        num_steps (int): Number of steps for LR range test.

    Returns:
        dict: Dictionary containing learning rates and corresponding losses.
    """
    model.train()

    def lr_lambda(step):
        return (max_lr / min_lr) ** (step / num_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    losses = []
    lrs = []
    best_loss = float("inf")

    progress_bar = tqdm.tqdm(dataloader, desc = "LR", leave = False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        if batch_idx > num_steps:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
        if batch_idx > 10 and loss.item() > 4 * best_loss:  # diverged
            logger.warning(f"Loss diverged at step {batch_idx} (loss={loss.item():.4f}, best_loss={best_loss:.4f}, lr={lr:.2e})")
            break

        progress_bar.set_postfix(lr=lr, loss=loss.item())

    min_idx = np.argmin(losses)
    best_lr = lrs[min_idx]
    suggested_lr = best_lr / 10

    logger.info(f"Suggested LR: {suggested_lr:.2e} (min loss LR: {best_lr:.2e})")

    return {"lrs": lrs, "losses": losses}

def main(
    data_dir: str,
    optimizer_type: str = "adam",
    criterion_type: str = "crossentropy",
    architecture: str = "resnet34",
    device: str = "auto",
    seed: int = 42,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    num_steps: int = 50,
    pretrained: bool = True,
    min_lr: float = 0.0000001,
    max_lr: float = 10
):
    """
    Run the Learning Rate Finder. Method is based from the paper 'Cyclical Learning Rates for Training Neural Networks (2017)'.

    Args:
        data_dir (str): Path to the dataset directory.
        optimizer_type (str): Optimizer type ("adam", "adamw", "sgd").
        criterion_type (str): Loss function type ("crossentropy", "bce", etc.).
        architecture (str): Model architecture.
        device (str): Computation device to use: "cpu", "cuda", or "auto".
        seed (int, optional): Random seed for reproducibility. Default is None.
        batch_size (int): Batch size for the dataloader.
        image_size (int): Image size.
        num_workers (int): Number of workers for data loading.
        num_steps (int): Number of steps to run during the LR test.
        pretrained (bool): Whether to load pretrained weights for the backbone.
        min_lr (float): Minimum learning rate for the LR range test.
        max_lr (float): Maximum learning rate for the LR range test.
    """
    if seed is not None:
        utils.set_seed(seed)
    device, use_cuda = utils.set_device(device)
    dataloader, num_classes = utils.dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_cuda=use_cuda
    )
    model = builder.build_model(
        num_of_classes=num_classes,
        architecture=architecture,
        pretrained=pretrained
    )
    model.to(device)
    optimizer = training_parameters.set_optimizer(
        model=model,
        lr=min_lr,
        optimizer_type=optimizer_type
    )
    criterion = training_parameters.set_criterion(
        criterion_type=criterion_type
    )

    lr_results = find_lr(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        min_lr=min_lr,
        max_lr=max_lr,
        num_steps=num_steps
    )

    plot_utils.plot_lr_finder(lr_results)