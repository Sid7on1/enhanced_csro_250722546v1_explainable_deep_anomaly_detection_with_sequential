import logging
import os
import tempfile
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from config import IMAGE_DIR, PREPROCESSED_DIR, PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing utilities for the computer vision project.

    This class provides functionality for loading, transforming, and augmenting images
    according to the project's requirements. It includes methods for common
    preprocessing techniques and can be extended with additional functionality as needed.

    ...

    Attributes
    ----------
    config : dict
        Preprocessing configuration, including parameters for resizing, augmentation, etc.

    Methods
    -------
    load_image(image_path)
        Load an image from the specified file path.
    preprocess_image(image, augment=False)
        Apply a series of preprocessing steps to the input image.
    resize_image(image, width, height)
        Resize the input image to the specified dimensions.
    augment_image(image)
        Apply data augmentation techniques to the input image.
    save_preprocessed_image(image, image_id)
        Save the preprocessed image to the specified directory.
    ...

    """

    def __init__(self, config: dict):
        self.config = config

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from the specified file path.

        Parameters
        ----------
        image_path : str
            File path of the image to be loaded

        Returns
        -------
        np.ndarray
            Loaded image as a numpy array

        Raises
        ------
        FileNotFoundError
            If the specified image file is not found
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    def preprocess_image(
        self, image: np.ndarray, augment: bool = False
    ) -> np.ndarray:
        """
        Apply a series of preprocessing steps to the input image.

        This method resizes the image, converts it to grayscale, and optionally applies
        data augmentation techniques. Additional preprocessing steps can be added as needed.

        Parameters
        ----------
        image : np.ndarray
            Input image to be preprocessed
        augment : bool, optional
            Flag indicating whether data augmentation should be applied, by default False

        Returns
        -------
        np.ndarray
            Preprocessed image

        Raises
        ------
        ValueError
            If the input image is not in the expected format or has invalid dimensions
        """
        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError("Invalid image format or dimensions.")

        image = self.resize_image(image, *self.config["resize_dims"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if augment:
            image = self.augment_image(image)

        return image

    def resize_image(
        self, image: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """
        Resize the input image to the specified dimensions.

        Parameters
        ----------
        image : np.ndarray
            Input image to be resized
        width : int
            Target width for the resized image
        height : int
            Target height for the resized image

        Returns
        -------
        np.ndarray
            Resized image

        """
        return cv2.resize(image, (width, height))

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation techniques to the input image.

        This method currently applies random horizontal flipping and rotation. Additional
        augmentation techniques can be added as needed.

        Parameters
        ----------
        image : np.ndarray
            Input image to be augmented

        Returns
        -------
        np.ndarray
            Augmented image

        """
        augment_config = self.config["augmentation"]

        if np.random.rand() < augment_config["flip_prob"]:
            image = cv2.flip(image, 1)

        angle = np.random.uniform(-augment_config["rotation_range"], augment_config["rotation_range"])
        image = Image.fromarray(image)
        image = image.rotate(angle, expand=True)

        return np.array(image)

    def save_preprocessed_image(
        self, image: np.ndarray, image_id: str, format: str = "png"
    ) -> str:
        """
        Save the preprocessed image to the specified directory.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed image to be saved
        image_id : str
            Unique identifier for the image
        format : str, optional
            File format for saving the image, by default "png"

        Returns
        -------
        str
            File path of the saved image

        """
        output_dir = os.path.join(PREPROCESSED_DIR, image_id)
        os.makedirs(output_dir, exist_ok=True)

        _, filename = tempfile.mkstemp(suffix=f".{format}")
        filename = os.path.join(output_dir, filename)

        cv2.imwrite(filename, image)
        logger.info(f"Saved preprocessed image: {filename}")

        return filename


class ImageDataset:
    """
    Class for managing a dataset of images and their corresponding preprocessed versions.

    This class provides functionality for loading, preprocessing, and iterating over images
    in a dataset. It also handles data augmentation and can be extended with additional
    functionality as needed.

    ...

    Attributes
    ----------
    image_paths : list of str
        List of file paths for the original images in the dataset
    preprocessed_paths : list of str
        List of file paths for the preprocessed images
    preprocessor : ImagePreprocessor
        Instance of the ImagePreprocessor class for applying preprocessing steps
    ...

    Methods
    -------
    load_dataset(image_dir)
        Load the dataset from the specified directory.
    preprocess_dataset(augment=False)
        Apply preprocessing to all images in the dataset.
    get_image(image_id)
        Retrieve the original and preprocessed image for a given image ID.
    ...

    """

    def __init__(self, preprocessor: ImagePreprocessor):
        self.image_paths = []
        self.preprocessed_paths = []
        self.preprocessor = preprocessor

    def load_dataset(self, image_dir: str) -> None:
        """
        Load the dataset from the specified directory.

        Parameters
        ----------
        image_dir : str
            Directory containing the original images

        """
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
        self.image_paths = sorted(image_paths)

    def preprocess_dataset(self, augment: bool = False) -> None:
        """
        Apply preprocessing to all images in the dataset.

        Parameters
        ----------
        augment : bool, optional
            Flag indicating whether data augmentation should be applied, by default False

        """
        for image_path in self.image_paths:
            image_id = os.path.basename(image_path)
            preprocessed_path = self.preprocessor.save_preprocessed_image(
                self.preprocessor.preprocess_image(self.preprocessor.load_image(image_path), augment),
                image_id,
            )
            self.preprocessed_paths.append(preprocessed_path)

    def get_image(
        self, image_id: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve the original and preprocessed image for a given image ID.

        Parameters
        ----------
        image_id : str
            Unique identifier for the image

        Returns
        -------
        tuple of np.ndarray, optional
            Original image and preprocessed image, or None if not found

        """
        original_image = None
        preprocessed_image = None

        for image_path in self.image_paths:
            if image_id in image_path:
                original_image = self.preprocessor.load_image(image_path)
                break

        for preprocessed_path in self.preprocessed_paths:
            if image_id in preprocessed_path:
                preprocessed_image = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
                break

        return original_image, preprocessed_image


def load_images(image_dir: str) -> List[np.ndarray]:
    """
    Load all images from the specified directory.

    Parameters
    ----------
    image_dir : str
        Directory containing the images

    Returns
    -------
    list of np.ndarray
        List of loaded images

    """
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images


def save_images(images: List[np.ndarray], output_dir: str, format: str = "png") -> None:
    """
    Save a list of images to the specified directory.

    Parameters
    ----------
    images : list of np.ndarray
        List of images to be saved
    output_dir : str
        Directory where the images will be saved
    format : str, optional
        File format for saving the images, by default "png"

    """
    os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(images):
        _, filename = tempfile.mkstemp(suffix=f".{format}")
        filename = os.path.join(output_dir, filename)
        cv2.imwrite(filename, image)
        logger.info(f"Saved image: {filename}")


def augment_images(
    images: List[np.ndarray],
    flip_prob: float = 0.5,
    rotation_range: int = 10,
) -> List[np.ndarray]:
    """
    Apply data augmentation techniques to a list of images.

    Parameters
    ----------
    images : list of np.ndarray
        List of images to be augmented
    flip_prob : float, optional
        Probability of applying horizontal flipping, by default 0.5
    rotation_range : int, optional
        Range of angles for random rotation, by default 10 degrees

    Returns
    -------
    list of np.ndarray
        List of augmented images

    """
    augmented_images = []

    for image in images:
        augmented_image = image.copy()

        if np.random.rand() < flip_prob:
            augmented_image = cv2.flip(augmented_image, 1)

        angle = np.random.uniform(-rotation_range, rotation_range)
        augmented_image = Image.fromarray(augmented_image)
        augmented_image = augmented_image.rotate(angle, expand=True)

        augmented_images.append(np.array(augmented_image))

    return augmented_images


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    parser.add_argument("--preprocessed_dir", type=str, default=PREPROCESSED_DIR)
    parser.add_argument("--config", type=str, default="preprocessing_config.json")
    args = parser.parse_args()

    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    # Load preprocessing configuration
    with open(args.config, "r") as f:
        PREPROCESSING_CONFIG = json.load(f)

    # Create image preprocessor
    preprocessor = ImagePreprocessor(PREPROCESSING_CONFIG)

    # Example: Preprocess a single image
    image_path = os.path.join(args.image_dir, "image1.jpg")
    preprocessed_image = preprocessor.preprocess_image(preprocessor.load_image(image_path))
    preprocessor.save_preprocessed_image(preprocessed_image, "image1")

    # Example: Preprocess a dataset
    dataset = ImageDataset(preprocessor)
    dataset.load_dataset(args.image_dir)
    dataset.preprocess_dataset(augment=True)

    # Retrieve preprocessed image
    original_image, preprocessed_image = dataset.get_image("image1.jpg")
    print(f"Original image shape: {original_image.shape}")
    print(f"Preprocessed image shape: {preprocessed_image.shape}")