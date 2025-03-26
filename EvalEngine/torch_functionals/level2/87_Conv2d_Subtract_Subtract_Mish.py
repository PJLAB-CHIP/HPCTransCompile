import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    subtract_value_1: float,
    subtract_value_2: float,
) -> torch.Tensor:
    """
    Applies convolution, subtracts two values, and applies Mish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Convolution bias tensor of shape (out_channels)
        subtract_value_1 (float): First value to subtract
        subtract_value_2 (float): Second value to subtract

    Returns:
        torch.Tensor: Output tensor after applying convolution, subtractions and Mish activation
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x - subtract_value_1
    x = x - subtract_value_2
    x = F.mish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.subtract_value_1,
            self.subtract_value_2,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]
