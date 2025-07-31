import logging
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# Constants and configuration
DATA_DIR = os.environ.get("DATA_DIR", "path/to/data")
IMG_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Exception classes
class DataLoadingException(Exception):
    pass

# Data structures/models
class ImageData:
    def __init__(self, path: str, label: int):
        self.path = path
        self.label = label

    def load_image(self) -> np.ndarray:
        try:
            image = Image.open(self.path)
            image = image.convert("RGB")
            image = image.resize(IMG_SIZE)
            image = np.array(image)
            return image
        except IOError as e:
            raise DataLoadingException(f"Error loading image {self.path}: {str(e)}")

# Main class with 10+ methods
class ImageDataset(Dataset):
    def __init__(self, data: List[ImageData], transform: Optional[Callable] = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.data[idx]
        image = example.load_image()

        if self.transform:
            image = self.transform(image)

        image = image.astype(np.float32)
        image = (image - MEAN) / STD
        image = image.transpose(2, 0, 1)

        return {"image": torch.from_numpy(image), "label": torch.tensor(example.label)}

# Helper classes and utilities
def create_dataloaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        raise DataLoadingException(f"Data file not found: {str(e)}")
    except pd.errors.EmptyDataError:
        raise DataLoadingException("Data file is empty.")
    except pd.errors.ParserError as e:
        raise DataLoadingException(f"Error parsing data file: {str(e)}")

# Validation functions
def validate_image_path(image_path: str) -> None:
    if not os.path.isfile(image_path):
        raise ValueError(f"Invalid image path: {image_path}")

# Utility methods
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Integration interfaces
def load_data(data_dir: str, file_name: str) -> pd.DataFrame:
    file_path = os.path.join(data_dir, file_name)
    df = load_data_from_csv(file_path)
    return df

def create_dataset(
    df: pd.DataFrame, transform: Optional[Callable] = None
) -> Tuple[Dataset, Dict[int, str]]:
    data = [
        ImageData(os.path.join(DATA_DIR, row["image_path"]), row["label"])
        for _, row in df.iterrows()
    ]

    unique_labels = df["label"].unique()
    label_map = {label: str(label) for label in unique_labels}

    dataset = ImageDataset(data, transform)
    return dataset, label_map

if __name__ == "__main__":
    # Example usage
    seed_everything(42)
    file_name = "images.csv"
    df = load_data(DATA_DIR, file_name)
    dataset, label_map = create_dataset(df)

    batch_size = 32
    num_workers = 4
    shuffle = True
    pin_memory = True

    dataloader = create_dataloaders(
        dataset, batch_size, num_workers, shuffle, pin_memory
    )

    for batch in dataloader:
        images = batch["image"]
        labels = batch["label"]
        print(images.shape, labels)