import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    instance_norm_weight: torch.Tensor,
    instance_norm_bias: torch.Tensor,
    divide_by: float,
) -> torch.Tensor:
    """
    Applies convolution, instance normalization and division by constant.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        instance_norm_weight (torch.Tensor): Instance norm weights
        instance_norm_bias (torch.Tensor): Instance norm bias
        divide_by (float): Constant to divide by

    Returns:
        torch.Tensor: Output tensor after convolution, normalization and division
    """
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.instance_norm(x, instance_norm_weight, instance_norm_bias)
    x = x / divide_by
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        instance_norm = nn.InstanceNorm2d(out_channels)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias
        self.instance_norm_weight = instance_norm.weight
        self.instance_norm_bias = instance_norm.bias
        self.divide_by = divide_by

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.instance_norm_weight,
            self.instance_norm_bias,
            self.divide_by,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divide_by = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
