import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    constant_value: float,
    scaling_factor: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, min with constant, bias addition and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        constant_value (float): Value to take minimum with
        scaling_factor (float): Factor to multiply output by
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        bias (torch.Tensor): Bias tensor to add of shape (out_channels, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution, min, bias and scaling
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = torch.min(x, torch.tensor(constant_value))
    x = x + bias
    x = x * scaling_factor
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant,
    adds a bias term, and multiplies by a scaling factor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        constant_value,
        bias_shape,
        scaling_factor,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias + torch.ones_like(conv.bias) * 0.02)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, constant_value, scaling_factor, fn=module_fn):
        return fn(
            x,
            constant_value,
            scaling_factor,
            self.conv_weight,
            self.conv_bias,
            self.bias,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, height, width),
        constant_value,
        scaling_factor,
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        constant_value,
        bias_shape,
        scaling_factor,
    ]
