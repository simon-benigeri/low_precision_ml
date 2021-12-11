import wandb
import numpy as np
from tqdm import tqdm

def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=1)

    # precision = config.precision
    # rounding = config.rounding
    # experiment_logs = f"{precision}-{rounding}" if rounding else f"{precision}"

    # Run training and track with wandb
    # total_batches = len(loader) * config.epochs
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
                # wandb.log({f"Train MSE": loss.item()}, step=step)
                # wandb.log({f"Train RMSE": np.sqrt(loss.item())}, step=step)

                # wandb.log({f"{experiment_logs} - Train MSE": loss.item()}, step=step)
                # wandb.log({f"{experiment_logs} - Train RMSE": np.sqrt(loss.item())}, step=step)
            step += 1
                # wandb.log({"Train MSE": loss.item()})
                # wandb.log({"Train RMSE": np.sqrt(loss.item())})

            # add loss for epoch loss
            # epoch_mse += loss.item()

        # epoch_mse /= len(loader.dataset)
        # epoch_rmse = np.sqrt(epoch_mse)

        wandb.log(
            {
                "Training": {
                    "MSE": loss.item(),
                    "RMSE": np.sqrt(loss.item()),
                    "epoch": epoch
                }
            }
        )
        """
        wandb.log(
            {
                "Training": {
                    "MSE": epoch_mse,
                    "RMSE": epoch_rmse,
                    "epoch": epoch
                }
            }
        )
        """

        # wandb.log({f"Train MSE - epoch": epoch_mse, 'epoch': epoch})
        # wandb.log({f"Train RMSE - epoch": epoch_rmse, 'epoch': epoch})
        # wandb.log({f"{experiment_logs} - Train MSE - epoch": epoch_mse}, step=epoch)
        # wandb.log({f"{experiment_logs} - Train RMSE - epoch": epoch_rmse}, step=epoch)
        """
        wandb.log({
            "Train MSE - epoch": epoch_mse,
            "Train RMSE- epoch": epoch_rmse})
        """



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
