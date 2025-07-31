import logging
import os
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Union
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluation:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_dataset(self, data_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        Load and preprocess the dataset.

        Parameters:
        - data_path (str): Path to the dataset folder.
        - batch_size (int): Number of samples per batch.
        - shuffle (bool): Whether to shuffle the dataset.

        Returns:
        - DataLoader: Loaded and preprocessed dataset.
        """
        # Define data transforms
        transform = self.transform

        # Load the dataset
        dataset = SewerDataset(root=data_path, transform=transform)

        # Create data loader
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

        return data_loader

    def evaluate(self, data_loader: DataLoader, classes: List[str], device: torch.device) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.

        Parameters:
        - data_loader (DataLoader): Data loader for the dataset.
        - classes (List[str]): List of class names/labels.
        - device (torch.device): Device to use for computation.

        Returns:
        - Dict[str, float]: Evaluation metrics and their values.
        """
        self.model.eval()
        correct = 0
        total = 0
        confusion_matrix_values = np.array(confusion_matrix(data_loader.dataset.labels, data_loader.dataset.predictions)))  # Fix this
        class_labels = data_loader.dataset.classes

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        precision, recall, f1_score, _ = self_calculate_metrics(confusion_matrix_values, class_labels)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        return metrics

    def calculate_metrics(self, confusion_matrix: np.array, class_labels: List[str]) -> Dict[str, float]:
        """
        Calculate evaluation metrics based on the confusion matrix.

        Parameters:
        - confusion_matrix (np.array): Confusion matrix of shape (num_classes, num_classes).
        - class_labels (List[str]): List of class labels.

        Returns:
        - Dict[str, float]: Evaluation metrics and their values.
        """
        # Fix the confusion_matrix calculation
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f_score = 2 * (precision * recall) / (precision + recall)

        # Fix the calculation for multiclass case
        precision = precision.sum() / len(class_labels)
        recall = recall.sum() / len(class_labels)
        f_score = f_score.sum() / len(class_labels)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f_score
        }

        return metrics

    def perform_evaluation(self, data_path: str, batch_size: int, shuffle: bool = False) -> Dict[str, float]:
        """
        Perform the complete evaluation process.

        Parameters:
        - data_path (str): Path to the dataset folder.
        - batch_size (int): Number of samples per batch.
        - shuffle (bool): Whether to shuffle the dataset.

        Returns:
        - Dict[str, float]: Evaluation metrics and their values.
        """
        # Load the dataset
        data_loader = self.load_dataset(data_path, batch_size, shuffle)

        # Get class labels
        classes = data_loader.dataset.classes

        # Get device for computation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Evaluate the model
        metrics = self.evaluate(data_loader, classes, device)

        return metrics

class SewerDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform=None):
        """
        Initializes the SewerDataset.

        Args:
            root (str): Root directory of the dataset.
            transform: Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        # Fix the following lines
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.predictions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.classes = ['normal', 'anomaly']

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        Args:
            idx (int): Index of the sample to be retrieved.

        Returns:
            tuple: (image, label) where 'image' is the preprocessed image and 'label' is the corresponding ground truth label.
        """
        # Get image path
        img_path = os.path.join(self.root, str(idx) + '.jpg')

        # Load and transform the image
        image = Image.open(img_path)
        image = self.transform(image)

        return image, self.labels[idx]

# Example usage
if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    eval_dataset_path = '/path/to/evaluation/dataset'
    batch_size = 32
    shuffle = True

    evaluator = Evaluation(model, device)
    metrics = evaluator.perform_evaluation(eval_dataset_path, batch_size, shuffle)

    logger.info(f'Evaluation results: {metrics}')