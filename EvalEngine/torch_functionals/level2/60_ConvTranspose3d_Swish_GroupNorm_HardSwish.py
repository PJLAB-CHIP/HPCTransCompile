import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    groups: int,
    eps: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, Swish activation, group normalization and HardSwish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        groups (int): Number of groups for group normalization
        eps (float): Epsilon value for group normalization
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        group_norm_weight (torch.Tensor): Weight tensor for group normalization
        group_norm_bias (torch.Tensor): Bias tensor for group normalization

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = torch.sigmoid(x) * x  # Swish activation
    x = F.group_norm(
        x, num_groups=groups, weight=group_norm_weight, bias=group_norm_bias, eps=eps
    )
    x = F.hardswish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation,
    group normalization, and then HardSwish activation.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups, eps
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)
        gn = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.group_norm_weight = nn.Parameter(gn.weight)
        self.group_norm_bias = nn.Parameter(gn.bias + torch.randn(out_channels) * 0.02)

    def forward(self, x, stride, padding, groups, eps, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            groups,
            eps,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.group_norm_weight,
            self.group_norm_bias,
        )


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, depth, height, width),
        stride,
        padding,
        groups,
        eps,
    ]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]
