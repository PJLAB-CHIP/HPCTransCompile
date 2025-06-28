import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """
    Applies convolution, scales the output, and performs minimum operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor
        conv_bias (torch.Tensor): Convolution bias tensor
        scale_factor (float): Scale factor to multiply output by

    Returns:
        torch.Tensor: Output tensor after convolution, scaling and min operation
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x * scale_factor
    x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum along channel dimension
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias + torch.ones_like(conv.bias) * 0.02)
        self.scale_factor = scale_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.scale_factor)


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
