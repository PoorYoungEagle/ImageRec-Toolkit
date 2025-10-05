"""
imagerec/__main__.py
Main entry point for the image classification framework.
"""
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

from imagerec.common import argument_parser
from imagerec.model_training import train
from imagerec.model_training import retrain
from imagerec.inference import infer, classify_folder
from imagerec.evaluation import evaluate
from imagerec.plot import plot
from imagerec.tools import split_dataset, limit_class_images, exporter, find_lr, compress


def main():
    try:
        arguments = argument_parser.main()

        if arguments.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        elif arguments.quiet:
            logging.getLogger().setLevel(logging.WARNING)

        logger.info("All arguments: ")
        for key, value in vars(arguments).items():
            logger.info(f"{key} : {value}")

        if arguments.command == "train":
            logger.info("Started training...")
            train.train_model(
                data_dir=arguments.data_dir,
                batch_size=arguments.batch_size,
                image_size=arguments.image_size,
                num_workers=arguments.num_workers,
                epochs=arguments.epochs,
                seed=arguments.seed,
                percentage=arguments.percentage,
                lr=arguments.lr,
                pretrained=arguments.pretrained,
                add_timestamp=arguments.add_timestamp,
                use_augments=arguments.use_augments,
                architecture=arguments.architecture,
                optimizer_type=arguments.optimizer_type,
                strategy=arguments.strategy,
                scheduler_type=arguments.scheduler_type,
                criterion_type=arguments.criterion_type,
                device=arguments.device,
                log_dir=arguments.log_dir,
                output_dir=arguments.output_dir,
                model_name=arguments.model_name,
                transforms_config=arguments.transforms_config
            )

        elif arguments.command == "retrain":
            logger.info("Started retraining model...")
            retrain.retrain_model(
                data_dir=arguments.data_dir,
                model_path=arguments.model_path,
                batch_size=arguments.batch_size,
                num_workers=arguments.num_workers,
                epochs=arguments.epochs,
                seed=arguments.seed,
                percentage=arguments.percentage,
                lr=arguments.lr,
                new_classes=arguments.new_classes,
                use_augments=arguments.use_augments,
                skip_missing_classes=arguments.skip_missing_classes,
                optimizer_type=arguments.optimizer_type,
                strategy=arguments.strategy,
                scheduler_type=arguments.scheduler_type,
                criterion_type=arguments.criterion_type,
                device=arguments.device,
                log_dir=arguments.log_dir,
                output_dir=arguments.output_dir,
                model_name=arguments.model_name,
                transforms_config=arguments.transforms_config
            )

        elif arguments.command == "infer":
            logger.info("Running inference...")
            infer.infer_class(
                model_path=arguments.model_path,
                input_path=arguments.input_path,
                device=arguments.device,
                top_n=arguments.top_n,
                include_classes=arguments.include_classes,
                exclude_classes=arguments.exclude_classes,
                config_labels=arguments.config_labels
            )

        elif arguments.command == "classify":
            logger.info("Started classifying folder...")
            classify_folder.classify(
                model_path=arguments.model_path,
                input_dir=arguments.input_dir,
                output_dir=arguments.output_dir,
                config_labels=arguments.config_labels,
                device=arguments.device,
                confidence_threshold=arguments.confidence_threshold,
                delete_original=arguments.delete_original
            )

        elif arguments.command == "evaluate":
            logger.info("Started evaluating...")
            evaluate.evaluate_model(
                model_path=arguments.model_path,
                data_dir=arguments.data_dir,
                evaluate_type=arguments.evaluate_type,
                device=arguments.device,
                batch_size=arguments.batch_size,
                num_workers=arguments.num_workers
            )

        elif arguments.command == "plot":
            logger.info("Started generating plots...")
            plot.plot_values(
                plot_command=arguments.plot_command,
                csv_path=getattr(arguments, "csv_path", None),
                image_path=getattr(arguments, "image_path", None),
                config_path=getattr(arguments, "config_path", None),
                visual_type=getattr(arguments, "visual_type", None),
                num_samples=getattr(arguments, "num_samples", None),
                image_size=getattr(arguments, "image_size", None)
            )

        elif arguments.command == "tools":
            if arguments.tool_command == "split":
                logger.info("Started splitting dataset...")
                split_dataset.split_dataset_by_class(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    split=arguments.split,
                    overwrite=arguments.overwrite
                )

            elif arguments.tool_command == "limit":
                logger.info("Started limiting class images...")
                limit_class_images.limit_class_images(
                    input_dir=arguments.input_dir,
                    output_dir=arguments.output_dir,
                    max_images=arguments.max_images,
                    mode=arguments.mode,
                    dry_run=arguments.dry_run
                )

            elif arguments.tool_command == "find-lr":
                logger.info("Started finding learning rate...")
                find_lr.main(
                    data_dir=arguments.data_dir,
                    optimizer_type=arguments.optimizer_type,
                    criterion_type=arguments.criterion_type,
                    architecture=arguments.architecture,
                    device=arguments.device,
                    seed=arguments.seed,
                    batch_size=arguments.batch_size,
                    image_size=arguments.image_size,
                    num_workers=arguments.num_workers,
                    num_steps=arguments.num_steps,
                    pretrained=arguments.pretrained,
                    min_lr=arguments.min_lr,
                    max_lr=arguments.max_lr
                )
            
            elif arguments.tool_command == "compress":
                logger.info("Started compressing model...")
                compress.compress_model(
                    model_path=arguments.model_path,
                    output_path=arguments.output_path,
                    method=arguments.method,
                    device=arguments.device,
                    amount=arguments.amount
                )

            elif arguments.tool_command == "export":
                if arguments.format == "onnx":
                    logger.info("Started exporting model to ONNX...")
                    exporter.export_to_onnx(
                        model_path=arguments.model_path,
                        output_path=arguments.output_path,
                        device=arguments.device,
                        opset_version=arguments.opset_version
                    )

                elif arguments.format == "torchscript":
                    logger.info("Started exporting model to TorchScript...")
                    exporter.export_to_torchscript(
                        model_path=arguments.model_path,
                        output_path=arguments.output_path,
                        device=arguments.device
                    )

                else:
                    logger.error("Invalid export format. Choose 'onnx' or 'torchscript'.")
                    sys.exit(1)

            else:
                logger.error("Unknown tool command. Use tools --help")
                sys.exit(1)

        else:
            logger.error("Unknown command. Use --help for available commands.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()