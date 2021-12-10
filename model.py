import torch.nn as nn
from torch import Tensor
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint, FixedPoint


class LinearLP(nn.Module):
    """
    a low precision Logistic Regression model
    """
    def __init__(self, input_dim: int = 13, quant: Quantizer = None):
        super(LinearLP, self).__init__()
        # architecture
        self.linear = nn.Linear(input_dim, 1)
        self.quant = quant

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        if self.quant:
            out = self.quant(out)
        return out
