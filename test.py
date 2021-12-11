import torch
import wandb
import numpy as np

def test(model, test_loader, criterion, config):
    model.eval()
    # precision = config.precision
    # rounding = config.rounding
    # experiment_logs = f"{precision}-{rounding}" if rounding else f"{precision}"

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

    wandb.log(
        { "Testing":
            {
                "MSE": avg_test_mse,
                "RMSE": avg_test_rmse
            }
        }
    )
    # wandb.log({f"Test RMSE": avg_test_rmse})
    # wandb.log({f"{experiment_logs} - Test MSE - epoch": avg_test_mse})
    # wandb.log({f"{experiment_logs} - Test RMSE - epoch": avg_test_rmse})

    """
    wandb.log({
        "Test MSE": avg_test_mse,
        "Test RMSE": avg_test_rmse})
    """
