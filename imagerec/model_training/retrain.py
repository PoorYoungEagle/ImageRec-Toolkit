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

def retrain_model(
    data_dir: str,
    model_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    epochs: int = 50,
    seed: int = 42,
    percentage: int = 100,
    lr: float = 0.001,
    new_classes: list = [],
    use_augments: bool = True,
    skip_missing_classes: bool = False,
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
    Executes the retraining pipeline for an existing ImageRec model.

    This function loads a previously trained model and adapts it to new classes or just retrains with more epochs.

    Args:
        data_dir (str): Path to dataset root.
        model_path (str): Path to the existing model checkpoint to retrain.
        batch_size (int): Number of images per batch.
        num_workers (int): Data loader workers.
        epochs (int): Number of training epochs.
        seed (int): Random seed.
        percentage (int): Percentage of available images per class to use during training (1-100). 100 uses all available data.
        lr (float): Initial learning rate.
        new_classes (list[str]): List of new class labels to add.
        use_augments (bool): Whether to apply data augmentation.
        skip_missing_classes (bool): If False, raises an error when there are missing classes. If True, it skips the missing classes and doesn't include them.
        optimizer_type (str): Optimizer type (e.g., "adam", "adamw", "sgd").
        strategy (str): Training strategy (e.g., "differential-lr").
        scheduler_type (str): Learning rate scheduler type.
        criterion_type (str): Loss function to use.
        device (str): Computation device ("cuda", "cpu", or "auto").
        log_dir (str): Directory to save training logs.
        output_dir (str): Directory to save retrained model checkpoints.
        model_name (str, optional): Name identifier for the retrained model.
        transforms_config (str): Path to custom transforms being applied to the images. (imagerec/configs/transforms/default.yaml)

    Outputs:
        - Retrained model checkpoint.
        - Backup of original model saved in backups directory.
        - Training logs written to log_dir.
        - Metrics printed.
        - Configuration file "config.json" saved
    """

    utils.set_seed(seed)
    device, is_cuda = utils.set_device(device) # device is a torch.device not a string

    model_values = builder.load_model_for_retraining(
        model_path=model_path,
        device=device,
        new_classes=new_classes,
        data_directory=data_dir
    )
    model = model_values["model"]
    merged_class_to_idx = model_values["class_to_idx"]
    new_classes = model_values["new_classes"]
    image_size = model_values["image_size"]
    pretrained = model_values["pretrained"]
    architecture = model_values["architecture"]

    train_loader, validation_loader = dataloader.get_retraining_dataloader(
        data_directory=data_dir,
        merged_class_to_idx=merged_class_to_idx,
        new_classes=new_classes,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        pin_memory=is_cuda,
        use_augments= use_augments,
        skip_missing_classes=skip_missing_classes,
        num_of_images_percentage=percentage
    )

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
    if criterion_type == "bce" and len(merged_class_to_idx) > 1:
        logger.warning(f"BCE only works with a single class. The data directory given contains {len(merged_class_to_idx)} classes. Reverting back to CrossEntropy.")
        criterion = training_parameters.set_criterion(
            criterion_type="crossentropy"
        )
    else:
        criterion = training_parameters.set_criterion(
            criterion_type=criterion_type
        )

    log_file, model_directory = train_utils.setup_retrain_paths(
        base_log_directory=log_dir,
        model_path=model_path,
        base_model_directory=output_dir,
        model_name=model_name
    )

    logger.debug(f"Optimizer state dict: {optimizer.state_dict()}")

    logger.info("Started Retraining")

    total_time_start = time.time()

    # it does not contain a best as it just retrains on the original model
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

    model_name = model_name or os.path.splitext(os.path.basename(model_path))[0] # uses model name if given or takes the name of the model
    save_path = os.path.join(model_directory, f"{model_name}.pt")

    train_utils.save_model(
        model=model,
        path=save_path,
        class_to_idx=merged_class_to_idx,
        architecture=architecture,
        pretrained=pretrained,
        optimizer=optimizer,
        epoch=epochs,
        image_size=image_size
    )

    logger.info(f"Saved last model\nValidation Loss: {validation_loss:.4f}")

    logger.info("Retraining Completed")

    total_time_stop = time.time()
    total_time_duration =  total_time_stop - total_time_start
    total_time_duration = datetime.timedelta(seconds=float(total_time_duration))

    logger.info(f"Total Duration: {total_time_duration}")
    logger.info(f"Model saved to: {os.path.join(model_directory, 'best.pt')}")
    logger.info(f"Training log saved to: {log_file}")

    class_names = list(merged_class_to_idx.keys())
    
    metrics.compute_metrics(model, validation_loader, class_names, device)

    config = {
        "data_dir" : data_dir,
        "model_path" : model_path,
        "batch_size" : batch_size,
        "num_workers" : num_workers,
        "epochs" : epochs,
        "seed" : seed,
        "percentage" : percentage,
        "lr" : lr,
        "new_classes" : new_classes,
        "use_augments" : use_augments,
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