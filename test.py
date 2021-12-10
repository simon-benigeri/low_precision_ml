import torch
import wandb
import numpy as np

def test(model, test_loader, criterion, config):
    model.eval()
    # Run the model on some test examples
    avg_test_mse = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # wandb.log({"test mse": loss.item()})
            # wandb.log({"test rmse": np.sqrt(loss.item())})
            avg_test_mse += loss.item()
    avg_test_mse /= len(test_loader.dataset)
    avg_test_rmse = np.sqrt(avg_test_mse)
    print('\nTest set: '
          'Average MSE: {:.4f}, '
          'Average RMSE: {:.4f})\n'.format(
        avg_test_mse, avg_test_rmse))
    wandb.log({
        "Test MSE": avg_test_mse,
        "Test RMSE": avg_test_rmse})