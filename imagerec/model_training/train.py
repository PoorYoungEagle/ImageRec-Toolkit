import os
import time
import datetime
import logging
logger = logging.getLogger(__name__)

from imagerec.model_training import dataloader
from imagerec.model_training import epoch_engine
from imagerec.model_training import train_utils
from imagerec.common import utils
from imagerec.common import training_parameters
from imagerec.common import builder
from imagerec.evaluation import metrics

def train_model(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    epochs: int = 50,
    seed: int = 42,
    percentage: int = 100,
    lr: float = 0.001,
    pretrained: bool = True,
    add_timestamp: bool = True,
    use_augments: bool = True,
    architecture: str = "resnet34",
    optimizer_type: str = "adam",
    strategy: str = "full",
    scheduler_type: str = "step",
    criterion_type: str = "crossentropy",
    device: str = "auto",
    log_dir: str = "imagerec/logs",
    output_dir: str = "imagerec/models",
    model_name: str = "model",
    transforms_config: str = "imagerec/configs/transforms/default.yaml"
):
    """
    Executes the full training and evaluation pipeline for ImageRec.

    Args:
        data_dir (str): Path to dataset root.
        batch_size (int): Number of images per batch.
        image_size (int): Target image resolution.
        num_workers (int): Data loader workers.
        epochs (int): Number of training epochs.
        seed (int): Random seed.
        percentage (int): Percentage of available images per class to use during training (1-100). 100 uses all available data.
        lr (float): Initial learning rate.
        pretrained (bool): Whether to use ImageNet-pretrained weights.
        add_timestamp (bool): Disable adding timestamp to output folder names.
        use_augments (bool): Whether to apply data augmentation.
        architecture (str): Model architecture to build.
        optimizer_type (str): Optimizer type ("adam", "adamw", "sgd").
        strategy (str): Training strategy ("full", "freeze_backbone", "differential-lr").
        scheduler_type (str): Learning rate scheduler type. ("step", "cosine", "plateau", "none")
        criterion_type (str): Loss function to use. ("crossentropy", "bce")
        device (str): Computation device ("auto", "cuda", "cpu").
        log_dir (str): Directory to save training logs.
        output_dir (str): Directory to save model checkpoints.
        model_name (str): Name identifier for the model.
        transforms_config (str): Path to custom transforms being applied to the images. (imagerec/configs/transforms/default.yaml)

    Outputs:
        - Model checkpoints: "best.pt" and "last.pt" inside output_dir.
        - Training logs written to log_dir.
        - Metrics printed.
        - Configuration file "config.json" saved
    """

    utils.set_seed(seed)
    device, is_cuda = utils.set_device(device) # device is a torch.device not a string

    train_loader, validation_loader, num_of_classes = dataloader.get_dataloader(
        data_directory=data_dir,
        transforms_config=transforms_config,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        num_of_images_percentage=percentage,
        use_augments=use_augments,
        pin_memory=is_cuda
    )
    model = builder.build_model(
        num_of_classes=num_of_classes,
        architecture=architecture,
        pretrained=pretrained
    )
    model = model.to(device)

    if batch_size > len(train_loader.dataset):
        batch_size = max(1, (len(train_loader.dataset) // 2))
        logger.warning(f"Batch size reduced to {batch_size} due to small dataset size.")


    optimizer = training_parameters.set_optimizer(
        model=model,
        lr=lr,
        optimizer_type=optimizer_type,
        strategy=strategy
    )
    scheduler = training_parameters.set_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        epochs=epochs
    )

    if criterion_type == "bce" and num_of_classes > 1:
        logger.warning(f"BCE only works with a single class. The data directory given contains {num_of_classes} classes. Reverting back to CrossEntropy.")
        criterion = training_parameters.set_criterion(
            criterion_type="crossentropy"
        )
    else:
        criterion = training_parameters.set_criterion(
            criterion_type=criterion_type
        )

    log_file, model_directory = train_utils.setup_train_paths(
        base_log_directory=log_dir,
        base_model_directory=output_dir,
        model_name=model_name,
        add_timestamp=add_timestamp
    )

    logger.debug(f"Optimizer state dict:\n {optimizer.state_dict()}")

    logger.info("Started Training")

    best_validation_loss = float("inf")
    total_time_start = time.time()

    for epoch in range(1, epochs+1):
        logger.info(f"Epoch: {epoch}/{epochs}")
        epoch_time_start = time.time()

        if criterion_type == "crossentropy":
            train_loss, train_accuracy = epoch_engine.train_epoch_ce(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
            validation_loss, validation_accuracy = epoch_engine.validate_epoch_ce(
                model=model,
                validation_loader=validation_loader,
                criterion=criterion,
                device=device
            )

        elif criterion_type == "bce":
            train_loss, train_accuracy = epoch_engine.train_epoch_bce(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
            validation_loss, validation_accuracy = epoch_engine.validate_epoch_bce(
                model=model,
                validation_loader=validation_loader,
                criterion=criterion,
                device=device
            )
        if scheduler_type in ["step", "cosine"]:
            scheduler.step()
        elif scheduler_type == "plateau":
            scheduler.step(validation_loss)

        epoch_time_stop = time.time()
        epoch_time_duration = epoch_time_stop - epoch_time_start
        epoch_time_duration = datetime.timedelta(seconds=float(epoch_time_duration))

        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")
        logger.info(f"Time for Epoch: {epoch_time_duration}")

        train_utils.model_logs(
            log_file=log_file,
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            learning_rate=optimizer.param_groups[0]['lr'],
            duration=epoch_time_duration
        )

        # this checks whether the number of epochs is atleast half the epochs placed in the cli, it only saves the best model after that 
        if epoch >= ((epochs + 1) // 2) and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            train_utils.save_model(
                model=model,
                path=os.path.join(model_directory, "best.pt"),
                class_to_idx=train_loader.dataset.class_to_idx,
                architecture=architecture,
                pretrained=pretrained,
                optimizer=optimizer,
                epoch=epoch,
                image_size=image_size
            )
            logger.info(f"Saved new best model\nValidation Loss: {validation_loss:.4f}")
    
    # saves model after its finished
    train_utils.save_model(
        model=model,
        path=os.path.join(model_directory, "last.pt"),
        class_to_idx=train_loader.dataset.class_to_idx,
        architecture=architecture,
        pretrained=pretrained,
        optimizer=optimizer,
        epoch=epoch,
        image_size=image_size
    )
    logger.info(f"Saved last model\nValidation Loss: {validation_loss:.4f}")

    logger.info("Training Completed")

    total_time_stop = time.time()
    total_time_duration =  total_time_stop - total_time_start
    total_time_duration = datetime.timedelta(seconds=float(total_time_duration))

    logger.info(f"Total Duration: {total_time_duration}")
    logger.info(f"Best validation loss: {best_validation_loss:.4f}")
    logger.info(f"Best model saved to: {os.path.join(model_directory, 'best.pt')}")
    logger.info(f"Training log saved to: {log_file}")

    class_names = list(train_loader.dataset.class_to_idx.keys())
    
    logger.debug(train_loader.dataset.class_to_idx)

    metrics.compute_metrics(model, validation_loader, class_names, device)

    config = {
        "data_dir" : data_dir,
        "batch_size" : batch_size,
        "image_size" : image_size,
        "num_workers" : num_workers,
        "epochs" : epochs,
        "seed" : seed,
        "percentage" : percentage,
        "lr" : lr,
        "pretrained" : pretrained,
        "use_augments" : use_augments,
        "architecture" : architecture,
        "optimizer_type" : optimizer_type,
        "strategy" : strategy,
        "scheduler_type" : scheduler_type,
        "criterion_type" : criterion_type,
        "log_dir" : log_dir,
        "output_dir" : output_dir,
        "model_name" : model_name,
        "transforms_config" : transforms_config,
        "total_time" : str(total_time_duration) # adding in an extra value to config
    }

    name = model_name + "_config"
    utils.save_config(
        config=config,
        config_name=name,
        path=model_directory
    )
    
    logger.info("Program Completed")