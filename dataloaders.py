from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_diabetes


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return features, target

class BostonDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        data = load_boston()
        X, y = data['data'], data['target']
        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.4, random_state=42)
        # Split the test data into a validation set and a test set
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        self.data_train = RegressionDataset(X_train, y_train)
        self.data_val = RegressionDataset(X_val, y_val)
        self.data_test = RegressionDataset(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

class DiabetesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        data = load_diabetes()
        X, y = data['data'], data['target']
        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.4, random_state=42)
        # Split the test data into a validation set and a test set
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        
        self.data_train = RegressionDataset(X_train, y_train)
        self.data_val = RegressionDataset(X_val, y_val)
        self.data_test = RegressionDataset(X_val, y_val)
        self.dims = tuple(self.data_train[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
