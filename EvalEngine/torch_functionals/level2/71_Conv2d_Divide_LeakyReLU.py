import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    """
    Applies convolution, division by constant, and LeakyReLU.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights of shape (out_channels, in_channels, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Convolution bias of shape (out_channels)
        divisor (float): Constant to divide by

    Returns:
        torch.Tensor: Output tensor after convolution, division and LeakyReLU activation
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x / divisor
    x = F.leaky_relu(x, negative_slope=0.01)
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.divisor = divisor

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.divisor)


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
