import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, GELU activation, and global average pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Convolution bias tensor of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels)
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = F.gelu(x)
    x = F.adaptive_avg_pool2d(x, 1)
    x = x.squeeze(-1).squeeze(-1)
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias)


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
