import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Applies convolution, bias addition, scaling, sigmoid activation and group normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor
        conv_bias (torch.Tensor): Convolution bias tensor
        bias (torch.Tensor): Bias tensor for addition
        scale (torch.Tensor): Scale tensor for multiplication
        group_norm_weight (torch.Tensor): Group norm weight tensor
        group_norm_bias (torch.Tensor): Group norm bias tensor
        num_groups (int): Number of groups for group normalization

    Returns:
        torch.Tensor: Output tensor after applying convolution, bias, scale, sigmoid and group norm
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x + bias
    x = x * scale
    x = torch.sigmoid(x)
    x = F.group_norm(x, num_groups, group_norm_weight, group_norm_bias)
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        bias_shape,
        scale_shape,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = nn.Parameter(
            conv.bias + torch.ones_like(conv.bias) * 0.02
        )  # make sure its nonzero
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.scale = nn.Parameter(torch.randn(scale_shape) * 0.02)
        group_norm = nn.GroupNorm(num_groups, out_channels)
        self.group_norm_weight = group_norm.weight
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.ones_like(group_norm.bias) * 0.02
        )  # make sure its nonzero

    def forward(self, x, num_groups, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.bias,
            self.scale,
            self.group_norm_weight,
            self.group_norm_bias,
            num_groups,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width), num_groups]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
