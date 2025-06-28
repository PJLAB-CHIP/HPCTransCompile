import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """Applies 3D convolution, softmax activation, and two max pooling operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Bias tensor for convolution of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor after applying convolution, softmax and max pooling,
            with shape (batch_size, out_channels, depth', height', width') where:
            depth' = ((depth - kernel_size + 1) // 4)
            height' = ((height - kernel_size + 1) // 4)
            width' = ((width - kernel_size + 1) // 4)
            The //4 comes from two max pooling operations with kernel_size=2
    """
    x = F.conv3d(x, conv_weight, conv_bias, stride=1, padding=0)
    x = F.softmax(x, dim=1)
    x = F.max_pool3d(x, kernel_size=2)
    x = F.max_pool3d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """

    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
