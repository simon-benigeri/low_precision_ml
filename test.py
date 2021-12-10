import torch
import wandb
import numpy as np

def test(model, test_loader, criterion, config):
    model.eval()
    # Run the model on some test examples
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        wandb.log({"test mse": loss.item()})
        wandb.log({"test rmse": np.sqrt(loss.item())})