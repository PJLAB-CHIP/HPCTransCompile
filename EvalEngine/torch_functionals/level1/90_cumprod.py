import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs a cumulative product operation.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to perform the cumulative product.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.cumprod(x, dim=dim)


class Model(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=module_fn):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return fn(x, self.dim)


# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1


def get_inputs():
    return [torch.randn(batch_size, *input_shape)]


def get_init_inputs():
    return [dim]
