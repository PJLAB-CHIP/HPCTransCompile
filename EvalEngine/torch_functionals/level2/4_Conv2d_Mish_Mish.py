import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Functional implementation of a sequence of operations:
    1. 2D convolution
    2. Mish activation
    3. Mish activation

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Weight tensor for convolution
        conv_bias (torch.Tensor): Bias tensor for convolution

    Returns:
        torch.Tensor: Output tensor after applying convolution and two Mish activations
    """
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.mish(x)
    x = F.mish(x)
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias)


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
