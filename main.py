import argparse
import wandb
import torch
import torch.nn as nn
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from qtorch import FloatingPoint, FixedPoint

from data import make_dataloader, get_data
from train import train
from test import test

from model import LinearLP


def make(config):
    # Make the data
    train, test = get_data(dataset=config.dataset, test_split=config.test_split)
    train_dataloader = make_dataloader(train, batch_size=config.batch_size)
    # val_dataloader = make_dataloader(val, batch_size=config.batch_size)
    test_dataloader = make_dataloader(test, batch_size=config.batch_size)

    # Make the quantizer
    # TODO: handle different permutations as user input. User config for this
    Q = None
    if config.quantize:
        bit_16 = FloatingPoint(exp=6, man=9)
        Q = Quantizer(forward_number=bit_16, backward_number=bit_16,
                      forward_rounding="nearest", backward_rounding="nearest")

    # Make the model
    model = LinearLP(input_dim=train.X[0].size, quant=Q).to(config.device)

    # Make the loss and optimizer
    criterion = nn.MSELoss()

    # Make the optimizer
    # TODO: handle different optimizers, like we saw with quantizer examples
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    return model, train_dataloader, test_dataloader, criterion, optimizer


def model_pipeline(config) -> nn.Module:
    # tell wandb to get started
    with wandb.init(project="low-precision-ml", entity="simonbenigeri", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_dataloader, test_dataloader, criterion, optimizer = make(config)
        # print(model)

        # and use them to train the model
        train(model, train_dataloader, criterion, optimizer, config)

        # and test its final performance
        test(model, test_dataloader, criterion, config)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a linear regression model.')

    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    # parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--quantize', type=bool, default=False)

    args = parser.parse_args()
    config = dict(
        device = "cuda" if torch.cuda.is_available() else "cpu",
        dataset = args.dataset,
        epochs = args.epochs,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        quantize = args.quantize,
        test_split= 0.4
    )
    model_pipeline(config)
