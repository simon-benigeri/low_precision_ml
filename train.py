import os
import argparse

from datetime import datetime
from pathlib import Path

from typing import Optional

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from pl_bolts.datamodules import SklearnDataModule
from sklearn.datasets import load_boston, load_diabetes

from model import LinearLP


def main(
        dataset: str = 'boston',
        gpus: Optional[int] = None,
        epochs: int = 1,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        batch_size: int = 32,
        num_workers: int = 1
    ):
    """[summary]

    Args:
        gpus (int, optional): [description]. Defaults to None.
        epochs (int, optional): [description]. Defaults to 1.
        learning_rate (float, optional): [description]. Defaults to 0.001.
        momentum (float, optional): [description]. Defaults to 0.9.
        batch_size (int, optional): [description]. Defaults to 32.
        num_workers (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    if dataset == 'boston':
        X, y = load_boston(return_X_y=True)
        input_dim = 13
    elif dataset == 'diabetes':
        input_dim = 10
    loaders = SklearnDataModule(X, y, val_split=0.2, test_split=0.1, random_state=42, batch_size=batch_size, num_workers=num_workers)
    
    model = LinearLP(input_dim=input_dim, dataset=dataset, learning_rate=learning_rate, momentum=momentum)
    trainer = Trainer(max_epochs=epochs, logger=WandbLogger(project="low-precision-ml", entity="simonbenigeri"))
    
    trainer.fit(model, train_dataloader=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())
    trainer.test(test_dataloaders=loaders.test_dataloader())

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This app showcases a dope CIFAR10 classifier')

    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=None)
    # parser.add_argument('--learning-rate', type=float, default=1)
    # parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=1)

    args = parser.parse_args()
    main(args.dataset, args.epochs)
