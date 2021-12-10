# low_precision_ml
 
## Setup

```
conda env create -f environment.yml  
conda activate low-precision-ml
```

## Running the training script
`python main.py --dataset diabetes --epochs 100 --learning-rate 0.01 --batch-size 64 `

Arguments:
* `dataset`: choose one of `'toy'`, `'diabetes'`, or `'boston'`
* `epochs`: number of epochs, default value is `5`
* `learning-rate`: learning rate for SGD, default value is `0.01`
* `batch-size`: batch size for minibatch SGD, default value is `256`

## Before you run the script
Go to `train.py` line 34. Make sure that in the following code line has **YOUR wandb project and entity info**.
```python
def model_pipeline(config) -> nn.Module:
    # tell wandb to get started
    with wandb.init(project="low-precision-ml", entity="simonbenigeri", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        ...
```