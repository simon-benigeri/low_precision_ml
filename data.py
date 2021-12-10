from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import StandardScaler

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

def make_dataloader(dataset: RegressionDataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=2)

def get_data(dataset: str='toy', test_split: float=0.4, val_split: float=0.5):
    if dataset == 'toy':
        # create dummy data for training
        x_values = [i for i in range(11)]
        X = np.array(x_values, dtype=np.float32)
        X = X.reshape(-1, 1)

        y_values = [2 * i + 1 for i in x_values]
        y = np.array(y_values, dtype=np.float32)
        # y_train = y_train.reshape(-1, 1)
    elif dataset == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
    else:
        X, y = load_boston(return_X_y=True)

    # standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # reshape y
    y = y.reshape(-1, 1)
    y = scaler.fit_transform(y)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=test_split)
    # Split the test data into a validation set and a test set
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_split)

    data_train = RegressionDataset(X_train, y_train)
    # data_val = RegressionDataset(X_val, y_val)
    data_test = RegressionDataset(X_test, y_test)

    return data_train, data_test

