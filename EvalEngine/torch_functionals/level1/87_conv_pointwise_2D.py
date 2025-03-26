import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs the pointwise 2D convolution using functional interface.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        weight (torch.Tensor): Weight tensor
        bias (torch.Tensor): Bias tensor

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
    """
    return F.conv2d(x, weight, bias=bias, stride=(1, 1), padding=(0, 0))


class Model(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            fn: Function to use for forward pass. Defaults to module_fn.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return fn(x, self.weight, self.bias)


# Constants
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, bias]
