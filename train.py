import wandb
import numpy as np
from tqdm import tqdm

def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=1)

    step = 0
    for epoch in tqdm(range(config.epochs)):
        epoch_mse = 0
        for batch_idx, (inputs, targets) in enumerate(loader):

            loss = train_batch(inputs, targets, model, optimizer, criterion, config)


            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(loader.dataset),
                           100. * batch_idx / len(loader), loss.item()))

                wandb.log(
                    {
                        "Training": {
                            "MSE": loss.item(),
                            "RMSE": np.sqrt(loss.item())
                        }
                    }, step=step
                )
            step += 1



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
