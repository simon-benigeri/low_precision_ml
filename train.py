import wandb
import numpy as np
from tqdm import tqdm

def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=1)

    # Run training and track with wandb
    # total_batches = len(loader) * config.epochs
    for epoch in tqdm(range(config.epochs)):
        for _, (inputs, targets) in enumerate(loader):

            loss = train_batch(inputs, targets, model, optimizer, criterion, config)
            wandb.log({"train mse loss": loss.item()})
            wandb.log({"train rmse": np.sqrt(loss.item())})



def train_batch(inputs, targets, model, optimizer, criterion, config):
    inputs, targets = inputs.to(config.device), targets.to(config.device)

    # Forward pass ➡
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss
