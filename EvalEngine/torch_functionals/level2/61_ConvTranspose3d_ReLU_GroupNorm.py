import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    groups: int,
    eps: float,
) -> torch.Tensor:
    """
    Applies a transposed 3D convolution, ReLU, and group normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        group_norm_weight (torch.Tensor): Weight tensor for group normalization
        group_norm_bias (torch.Tensor): Bias tensor for group normalization
        groups (int): Number of groups for group normalization
        eps (float): Epsilon for group normalization
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W)
    """
    x = F.conv_transpose3d(x, conv_transpose, bias=None)
    x = F.relu(x)
    x = F.group_norm(x, groups, group_norm_weight, group_norm_bias, eps)
    return x


class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups, bias):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_transpose_parameter = conv.weight

        # set torch seed to 0
        torch.manual_seed(0)
        gn = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.group_norm_weight = nn.Parameter(
            gn.weight + torch.randn_like(gn.weight) * 0.02
        )
        self.group_norm_bias = nn.Parameter(gn.bias + torch.randn_like(gn.bias) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_transpose_parameter,
            self.group_norm_weight,
            self.group_norm_bias,
            groups,
            eps,
        )


batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
