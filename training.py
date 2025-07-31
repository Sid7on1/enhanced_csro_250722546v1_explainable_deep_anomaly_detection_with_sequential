import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from typing import List, Tuple, Dict

# Constants
VELOCITY_THRESHOLD = 0.1  # From paper
FLOW_THRESHOLD = 0.2  # From paper

class AnomalyDataset(Dataset):
    def __init__(self, data_path: str, transform: transforms.Compose = None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        image_data = torch.tensor(row['image_data'].values, dtype=torch.float32)
        label = row['anomaly']  # 0 for normal, 1 for anomaly

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, label

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = AnomalyDataset(config['train_data_path'], transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        self.test_dataset = AnomalyDataset(config['test_data_path'])
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config['batch_size'])
        self.model = InceptionV3(pretrained=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self) -> None:
        for epoch in range(self.config['epochs']):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def test(self) -> None:
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target.unsqueeze(1)).item()  # sum up batch loss
                pred = torch.round(output)  # round-off predictions
                correct += (pred == target).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

# Example usage
if __name__ == '__main__':
    config = {
        'train_data_path': 'train_data.csv',
        'test_data_path': 'test_data.csv',
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001
    }
    trainer = Trainer(config)
    trainer.train()
    trainer.test()
    trainer.save_model('model.pth')