import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    eps: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """
    Applies convolution, group normalization, tanh, hardswish, residual addition and logsumexp.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        eps (float): Small constant for numerical stability in group norm
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        group_norm_weight (torch.Tensor): Group norm weights
        group_norm_bias (torch.Tensor): Group norm bias
        groups (int): Number of groups for group norm

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    # Convolution
    x_conv = F.conv2d(x, conv_weight, conv_bias)

    # Group Normalization
    x_norm = F.group_norm(x_conv, groups, group_norm_weight, group_norm_bias, eps)

    # Tanh
    x_tanh = torch.tanh(x_norm)

    # HardSwish
    x_hard_swish = F.hardswish(x_tanh)

    # Residual Addition
    x_res = x_conv + x_hard_swish

    # LogSumExp
    x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)

    return x_logsumexp


class Model(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish,
    Residual Addition, and LogSumExp.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups, eps):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias
        group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.group_norm_weight = group_norm.weight
        self.group_norm_bias = group_norm.bias

    def forward(self, x, eps, groups, fn=module_fn):
        return fn(
            x,
            eps,
            self.conv_weight,
            self.conv_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            groups,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
groups = 8
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width), eps, groups]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, eps]
