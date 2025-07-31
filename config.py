import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants and configuration
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD = [0.229, 0.224, 0.225]
MODEL_EMBEDDING_SIZE = 1024
MODEL_NUM_CLASSES = 2

TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
    ]
)

TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
    ]
)


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict
        self.load_config()

    def load_config(self):
        self.data_dir = self.get_config_value("data_dir")
        self.model_name = self.get_config_value("model_name")
        self.model_path = self.get_config_value("model_path")
        self.device = self.get_config_value("device")
        self.num_classes = self.get_config_value("num_classes")
        self.image_size = self.get_config_value("image_size")
        self.batch_size = self.get_config_value("batch_size")
        self.num_workers = self.get_config_ |
class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize the configuration object.

        Parameters:
            config_dict (Dict[str, Any]): Dictionary containing configuration settings.
        """
        self.config_dict = config_dict
        self.load_config()

    def load_config(self):
        """
        Load configuration settings from the dictionary.
        """
        self.data_dir = self.get_config_value("data_dir")
        self.model_name = self.get_config_value("model_name")
        self.model_path = self.get_config_value("model_path")
        self.device = self.get_config_value("device")
        self.num_classes = self.get_config_value("num_classes")
        self.image_size = self.get_config_value("image_size")
        self.batch_size = self.get_config_value("batch_size")
        self.num_workers = self.get_config_value("num_workers")
        self.learning_rate = self.get_config_value("learning_rate")
        self.momentum = self.get_config_value("momentum")
        self.weight_decay = self.get_config_value("weight_decay)
        self.epochs = self.get_config_value("epochs")
        self.checkpoint_dir = self.get_config_value("checkpoint_dir")
        self._validate_config()

    def get_config_value(self, key: str) -> Any:
        """
        Retrieve a configuration value from the dictionary.

        Parameters:
            key (str): Configuration key to retrieve.

        Returns:
            Any: Value associated with the given key.
        """
        if key not in self.config_dict:
            raise KeyError(f"Missing configuration key: {key}")
        return self.config_dict[key]

    def _validate_config(self):
        """
        Validate the configuration settings. Raise errors for missing or invalid values.
        """
        required_keys = [
            "data_dir",
            "model_name",
            "model_path",
            "device",
            "num_classes",
            "image_size",
            "batch_size",
            "num_workers",
            "learning_rate",
            "momentum",
            "weight_decay",
            "epochs",
            "checkpoint_dir",
        ]
        for key in required_keys:
            if key not in self.config_dict:
                raise KeyError(f"Missing required configuration key: {key}")

        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Invalid data directory: {self.data_dir}")

        if not torch.cuda.is_available() and self.device == "cuda":
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            self.device = "cpu"

        if not isinstance(self.num_classes, int) or self.num_classes <= 0:
            raise ValueError("Invalid number of classes. Must be a positive integer.")

        if not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError("Invalid image size. Must be a positive integer.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Invalid batch size. Must be a positive integer.")

        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("Invalid number of workers. Must be a non-negative integer.")

        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate. Must be a positive float.")

        if not isinstance(self.momentum, float) or not 0 <= self.momentum <= 1:
            raise ValueError("Invalid momentum value. Must be between 0 and 1.")

        if not isinstance(self.weight_decay, float) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay. Must be a non-negative float.")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("Invalid number of epochs. Must be a positive integer.")

        if not os.path.isdir(self.checkpoint_dir):
            raise ValueError(f"Invalid checkpoint directory: {self.checkpoint_dir}")

    @property
    def transform_train(self):
        """
        Return the data transformation pipeline for training.
        """
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
            ]
        )

    @property
    def transform_test(self):
        """
        Return the data transformation pipeline for testing.
        """
        return transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
            ]
        )

    def get_model(self) -> nn.Module:
        """
        Return the model instance based on the configuration.
        """
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == "resnet101":
            model = models.resnet101(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        model.to(self.device)
        return model

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Return the optimizer based on the configuration.

        Parameters:
            model (nn.Module): Model to optimize.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        return torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_samples: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Return the learning rate scheduler based on the configuration.

        Parameters:
            optimizer (torch.optim.Optimizer): Optimizer to use.
            num_training_samples (int): Number of training samples.

        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: Learning rate scheduler instance, or None.
        """
        if "step_size" in self.config_dict:
            step_size = self.get_config_value("step_size")
            gamma = self.get_config_value("gamma")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
            return scheduler
        return None

    def get_dataloaders(
        self, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset
    ) -> Dict[str, DataLoader]:
        """
        Return DataLoaders for training and validation based on the configuration.

        Parameters:
            train_dataset (torch.utils.data.Dataset): Training dataset.
            val_dataset (torch.utils.data.Dataset): Validation dataset.

        Returns:
            Dict[str, DataLoader]: Dictionary containing "train" and "val" DataLoaders.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return {"train": train_loader, "val": val_loader}


def load_config(config_file: str) -> Config:
    """
    Load configuration settings from a file and return a Config object.

    Parameters:
        config_file (str): Path to the configuration file.

    Returns:
        Config: Configuration object.
    """
    import yaml

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def save_config(config: Config, config_file: str):
    """
    Save the configuration settings to a file.

    Parameters:
        config (Config): Configuration object to save.
        config_file (str): Path to the configuration file.
    """
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(config.config_dict, f, default_flow_style=False)


def update_config(config: Config, updates: Dict[str, Any]):
    """
    Update the configuration settings with the provided updates.

    Parameters:
        config (Config): Configuration object to update.
        updates (Dict[str, Any]): Dictionary containing updated settings.
    """
    config.config_dict.update(updates)
    config.load_config()


def log_config(config: Config):
    """
    Log the configuration settings.

    Parameters:
        config (Config): Configuration object to log.
    """
    logger.info("Configuration settings:")
    for key, value in config.config_dict.items():
        logger.info(f"{key}: {value}")


# Example usage
if __name__ == "__main__":
    config_dict = {
        "data_dir": "/path/to/data",
        "model_name": "resnet50",
        "model_path": "models/model.pth",
        "device": "cuda",
        "num_classes": 2,
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 4,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "epochs": 100,
        "checkpoint_dir": "checkpoints",
    }

    config = Config(config_dict)
    log_config(config)

    # Example of using the configuration
    model = config.get_model()
    optimizer = config.get_optimizer(model)
    scheduler = config.get_scheduler(optimizer, num_training_samples=1000)  # Replace with actual value
    dataloaders = config.get_dataloaders(
        train_dataset=None, val_dataset=None
    )  # Replace with actual datasets

    # ... continue with training, etc.