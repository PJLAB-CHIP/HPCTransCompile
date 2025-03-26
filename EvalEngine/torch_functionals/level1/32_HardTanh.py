import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Applies HardTanh activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        min_val (float): The minimum value for the HardTanh function.
        max_val (float): The maximum value for the HardTanh function.

    Returns:
        torch.Tensor: Output tensor with HardTanh applied, same shape as input.
    """
    return F.hardtanh(x, min_val=min_val, max_val=max_val)


class Model(nn.Module):
    """
    Simple model that performs a HardTanh activation.
    """

    def __init__(self, min_val, max_val):
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return fn(x, self.min_val, self.max_val)


batch_size = 16
dim = 16384
min_val = -1.0
max_val = 1.0


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return [min_val, max_val]
