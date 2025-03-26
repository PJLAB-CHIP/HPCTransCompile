import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    dim: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, minimum operation along specified dimension, and softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        dim (int): Dimension along which to apply minimum operation
        conv_weight (torch.Tensor): 3D convolution weight tensor
        conv_bias (torch.Tensor): 3D convolution bias tensor

    Returns:
        torch.Tensor: Output tensor after applying convolution, min and softmax
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = torch.min(x, dim=dim)[0]  # Apply minimum along the specified dimension
    x = F.softmax(x, dim=1)  # Apply softmax along the channel dimension
    return x


class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension,
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = nn.Parameter(
            conv.bias + torch.ones_like(conv.bias) * 0.02
        )  # make sure its nonzero
        self.dim = dim

    def forward(self, x, fn=module_fn):
        return fn(x, self.dim, self.conv_weight, self.conv_bias)


batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)


def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
