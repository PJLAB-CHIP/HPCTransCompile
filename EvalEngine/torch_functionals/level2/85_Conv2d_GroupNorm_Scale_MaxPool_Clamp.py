import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    scale: torch.Tensor,
    num_groups: int,
    maxpool_kernel_size: int,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """
    Applies convolution, group normalization, scaling, max pooling and clamping.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        group_norm_weight (torch.Tensor): Group norm weights
        group_norm_bias (torch.Tensor): Group norm bias
        scale (torch.Tensor): Scale parameter of shape (out_channels, 1, 1)
        num_groups (int): Number of groups for group norm
        maxpool_kernel_size (int): Kernel size for max pooling
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = F.group_norm(x, num_groups, weight=group_norm_weight, bias=group_norm_bias)
    x = x * scale
    x = F.max_pool2d(x, kernel_size=maxpool_kernel_size)
    x = torch.clamp(x, clamp_min, clamp_max)
    return x


class Model(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        group_norm = nn.GroupNorm(num_groups, out_channels)
        self.group_norm_weight = nn.Parameter(
            group_norm.weight + torch.randn(group_norm.weight.shape) * 0.02
        )
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.randn(group_norm.bias.shape) * 0.02
        )
        self.scale = nn.Parameter(torch.randn(scale_shape) * 0.02)
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.scale,
            self.num_groups,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ]
