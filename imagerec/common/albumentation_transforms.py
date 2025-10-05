import albumentations as A
import numpy as np
import PIL.Image
import yaml
import os
import logging
logger = logging.getLogger(__name__)

AUGMENT_MAP = {
    "random_rain" : A.RandomRain,
    "random_fog" : A.RandomFog,
    "random_sun_flare" : A.RandomSunFlare,
    "random_shadow" : A.RandomShadow,
    "color_jitter" : A.ColorJitter,
    "hue_saturation_value" : A.HueSaturationValue,
    "rgb_shift" : A.RGBShift,
    "channel_shuffle" : A.ChannelShuffle,
    "coarse_dropout" : A.CoarseDropout,
    "grid_dropout" : A.GridDropout,
    "elastic_transform" : A.ElasticTransform,
    "grid_distortion" : A.GridDistortion,
    "optical_distortion" : A.OpticalDistortion,
    "motion_blur" : A.MotionBlur,
    "gauss_noise" : A.GaussNoise,
    "gaussian_blur" : A.GaussianBlur,
    "random_resized_crop" : A.RandomResizedCrop,
    "perspective" : A.Perspective,
    "horizontal_flip" : A.HorizontalFlip,
    "rotate" : A.Rotate,
    "random_brightness_contrast" : A.RandomBrightnessContrast
}


class TrainTransform:
    """
    Training data transformation pipeline using the library "Albumentations".
    Loads augmentation configuration from a YAML file (if available) or
    falls back to default augmentations.

    Args:
        image_size (int): Target image size (height = width). Default is 224.
        config_path (str): Path to the YAML config file for augmentations.
        is_training (bool): Whether to apply training augmentations.
        use_augments (bool): Whether to enable augmentations at all.
    """
    def __init__(
        self,
        image_size: int = 224,
        config_path="configs/transforms/default.yaml",
        is_training: bool = True,
        use_augments: bool = True
    ):
        self.image_size = image_size
        self.is_training = is_training
        self.use_augments = use_augments
        self.config_path = config_path
        self.transform = self._set_transforms()

    def _load_config(self, path):
        """Load YAML augmentation config if available, else return None."""
        if not path:
            logger.warning(f"YAML file not given.\nUsing hard coded default values.")
            return None
        if os.path.isdir(path):
            logger.error(f"Please select the .yaml file: {path}")
            raise FileNotFoundError(f"Please select the .yaml file: {path}")
        
        if not os.path.isabs(path):
            script_dir = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(script_dir, path)

        if not os.path.exists(path):
            logger.warning(f"YAML file not found at {path}\nUsing hard coded default values.")
            return None

        with open(path, 'r') as file:
            logger.info("Loading transform config...")
            return yaml.safe_load(file)

    def _set_transforms(self):
        """Build Albumentations transform pipeline based on config or defaults."""

        transforms = []
        if self.is_training and self.use_augments:
            config = self._load_config(self.config_path)
            if config:
                augments_config = config["augmentations"]
                # for oneof groups
                oneof_groups = augments_config.get("oneof_groups", {})
                for group_name, group in oneof_groups.items():
                    if group.get("enabled", False):
                        oneof_transforms = []

                        for name, params in group.get("transforms", {}).items():
                            if not params.get("enabled", False):
                                continue
                            
                            albumentation_class = AUGMENT_MAP.get(name.lower())
                            if albumentation_class:
                                correct_params = {key: value for key, value in params.items() if key != "enabled"}
                                oneof_transforms.append(albumentation_class(**correct_params))
                            else:
                                print(f"Warning: '{name}' not found in AUGMENT_MAP.")

                        if oneof_transforms:
                            transforms.append(A.OneOf(oneof_transforms, p=group.get("p", 0.3)))
                
                # for standalone augments
                for name, params in augments_config.items():
                    if name == "oneof_groups" or not params.get("enabled", False):
                        continue

                    albumentation_class = AUGMENT_MAP.get(name.lower())
                    if albumentation_class:
                        correct_params = {key: value for key, value in params.items() if key != "enabled"}
                        if name.lower() == "random_resized_crop":
                            correct_params["size"] = (self.image_size, self.image_size)
                        transforms.append(albumentation_class(**correct_params))
                    else:
                        print(f"Warning: '{name}' not found in AUGMENT_MAP.")
            else:
                logger.info("Using hardcoded default augmentations.")
                transforms = [
                    A.OneOf([
                        A.RandomRain(blur_value=2),
                        A.RandomFog(fog_coef_range=(0.2, 0.5)),
                        A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9)),
                        A.RandomShadow(shadow_dimension=5)
                    ], p=0.3),
                    A.OneOf([
                        A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                        A.HueSaturationValue(),
                        A.RGBShift(),
                        A.ChannelShuffle()
                    ]),
                    A.OneOf([
                        A.CoarseDropout(),
                        A.GridDropout()
                    ]),
                    A.OneOf([
                        A.ElasticTransform(),
                        A.GridDistortion(),
                        A.OpticalDistortion()
                    ]),
                    A.OneOf([
                        A.MotionBlur(blur_limit=7),
                        A.GaussNoise(std_range=(0.1, 0.2)),
                        A.GaussianBlur(blur_limit=(3, 5))
                    ], p=0.3),
                    A.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.8, 1.0)),
                    A.Perspective(scale=(0.05, 0.10), keep_size=True, p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ]

        # must include these augments
        transforms.append(A.Resize(height=self.image_size, width=self.image_size))
        transforms.append(A.Normalize())
        transforms.append(A.pytorch.ToTensorV2())

        return A.Compose(transforms)

    def __call__(self, image):
        """
        Apply training transforms.

        Args:
            image (PIL.Image.Image or np.ndarray): Input image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        image = np.array(image)
        image = self.transform(image=image)["image"]
        return image


class InferenceTransform:
    """
    Inference transform pipeline for evaluation or deployment.

    Applies resize, normalization, and tensor conversion without
    data augmentations.

    Args:
        image_size (int): Target image size. Default is 224.
    """
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = self._set_transforms()

    def _set_transforms(self):
        """Build inference transforms."""
        return A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(),
            A.pytorch.ToTensorV2()
        ])

    def __call__(self, image: str):
        """
        Apply inference transforms.

        Args:
            image (str): File path

        Returns:
            torch.Tensor: Transformed image tensor with batch dimension.
        """
        if isinstance(image, str):
            image = PIL.Image.open(image).convert('RGB')
        else:
            logger.error("Expected image to be a file path")
            raise TypeError("Expected image to be a file path")

        image = np.array(image)
        image = self.transform(image=image)["image"]
        image = image.unsqueeze(0)
        
        return image