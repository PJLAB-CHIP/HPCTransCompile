import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    batch_norm_weight: torch.Tensor,
    batch_norm_bias: torch.Tensor,
    batch_norm_running_mean: torch.Tensor,
    batch_norm_running_var: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Applies transposed convolution, batch norm, tanh, max pool and group norm operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        conv_transpose (torch.Tensor): Transposed conv weights
        conv_transpose_bias (torch.Tensor): Transposed conv bias
        batch_norm_weight (torch.Tensor): BatchNorm weight parameter
        batch_norm_bias (torch.Tensor): BatchNorm bias parameter
        batch_norm_running_mean (torch.Tensor): BatchNorm running mean
        batch_norm_running_var (torch.Tensor): BatchNorm running variance
        group_norm_weight (torch.Tensor): GroupNorm weight parameter
        group_norm_bias (torch.Tensor): GroupNorm bias parameter
        num_groups (int): Number of groups for group norm

    Returns:
        torch.Tensor: Output after applying all operations
    """
    x = F.conv_transpose2d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = F.batch_norm(
        x,
        batch_norm_running_mean,
        batch_norm_running_var,
        batch_norm_weight,
        batch_norm_bias,
        training=True,
    )
    x = torch.tanh(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    x = F.group_norm(
        x, num_groups=num_groups, weight=group_norm_weight, bias=group_norm_bias
    )
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation,
    max pooling, and group normalization.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        num_groups,
    ):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.conv_transpose_weight = self.conv_transpose.weight
        self.conv_transpose_bias = self.conv_transpose.bias

        self.batch_norm_weight = self.batch_norm.weight
        self.batch_norm_bias = self.batch_norm.bias
        self.register_buffer("batch_norm_running_mean", self.batch_norm.running_mean)
        self.register_buffer("batch_norm_running_var", self.batch_norm.running_var)

        self.group_norm_weight = self.group_norm.weight
        self.group_norm_bias = self.group_norm.bias

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.batch_norm_weight,
            self.batch_norm_bias,
            self.batch_norm_running_mean,
            self.batch_norm_running_var,
            self.group_norm_weight,
            self.group_norm_bias,
            num_groups,
        )


batch_size = 128
in_channels = 32
out_channels = 64
kernel_size = 4
stride = 2
padding = 1
groups = 8
num_groups = 4
height, width = 32, 32


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
