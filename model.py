import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
import pytorch_lightning as pl
import torchmetrics

from torchmetrics import MeanSquaredError, MeanSquaredLogError, R2Score, MeanAbsoluteError


from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint, FixedPoint



class LinearLP(pl.LightningModule):
    """
    a low precision Logistic Regression model
    """
    def __init__(self, learning_rate: float, momentum: float, input_dim: int=13, dataset: str='boston', quant: Quantizer=None):
        super(LinearLP, self).__init__()
        self.dataset = dataset
        # This saves all constructor arguments as items in the hparams dictionary
        self.save_hyperparameters()

        # architecture
        self.linear = nn.Linear(input_dim, 1)
        self.quant = quant

        # metrics
        # self.train_mse = MeanSquaredError()
        # self.train_msle = MeanSquaredLogError()
        # self.train_mae = MeanAbsoluteError()
        # self.train_r2_score = R2Score()

        # self.val_mse = MeanSquaredError()
        # self.val_msle = MeanSquaredLogError()
        # self.val_mae = MeanAbsoluteError()
        # self.val_r2_score = R2Score()

        # self.test_mse = MeanSquaredError()
        # self.test_msle = MeanSquaredLogError()
        # self.test_mae = MeanAbsoluteError()
        # self.test_r2_score = R2Score()
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        if self.quant:
            out = self.quant(out)

        return out

    @staticmethod
    def mse_loss(outputs, targets):
        return F.mse_loss(outputs, targets)
        # return nn.MSELoss(outputs, targets)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, targets = batch
        targets = targets.unsqueeze(-1)
        outputs = self.forward(inputs)

        loss = self.mse_loss(outputs, targets)
        self.log("train loss", loss, on_step=True, on_epoch=True)

        # self.log("train performance", {
            # "train_mse": self.train_mse(outputs, targets)# ,
            # "train_msle": self.train_msle(outputs, targets),
            # "train_mae": self.train_mae(outputs, targets),
            # "train_r2": self.train_r2_score(outputs, targets)
        # }, on_step=True, on_epoch=True)

        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        pass
        # self.log('train_mse_epoch', self.train_mse.compute())
        # self.log('train_msle_epoch', self.train_msle.compute())
        # self.log('train_mae_epoch', self.train_mae.compute())
        # self.log('train_r2_epoch', self.train_r2_score.compute())

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.unsqueeze(-1)
        outputs = self.forward(inputs)
        
        loss = self.mse_loss(outputs, targets)
        self.log("val loss", loss, on_step=True, on_epoch=True)

        # self.log("val performance", {
            # "val_mse": self.val_mse(outputs, targets)# ,
            # "val_msle": self.val_msle(outputs, targets),
            # "val_mae": self.val_mae(outputs, targets),
            # "val_r2": self.val_r2_score(outputs, targets)
        # }, on_step=True, on_epoch=True)
        
        return loss

    def validation_epoch_end(self, outs):
        # log epoch metric
        pass
        # self.log('val_mse_epoch', self.val_mse.compute())
        # self.log('val_msle_epoch', self.val_msle.compute())
        # self.log('val_mae_epoch', self.val_mae.compute())
        # self.log('val_r2_epoch', self.val_r2_score.compute())

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.unsqueeze(-1)
        outputs = self.forward(inputs)
        
        loss = self.mse_loss(outputs, targets)
        self.log("test loss", loss, on_step=True, on_epoch=True)

        # self.log("test performance", {
            # "test_mse": self.test_mse(outputs, targets)# ,
            # "test_msle": self.test_msle(outputs, targets),
            # "test_mae": self.test_mae(outputs, targets),
            # "test_r2": self.test_r2_score(outputs, targets)
        # }, on_step=True, on_epoch=True)
        
        return loss

    def test_epoch_end(self, outputs):
        return super().test_epoch_end(outputs)

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
