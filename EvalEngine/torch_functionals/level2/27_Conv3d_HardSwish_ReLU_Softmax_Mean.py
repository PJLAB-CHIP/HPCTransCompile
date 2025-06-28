import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, HardSwish, ReLU, Softmax and mean reduction.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): 3D convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Bias tensor for 3D convolution of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor after applying convolution, activations and reduction,
            with shape (batch_size, out_channels)
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = F.hardswish(x)
    x = F.relu(x)
    x = F.softmax(x, dim=1)
    x = torch.mean(x, dim=[2, 3, 4])
    return x


class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies HardSwish, ReLU, Softmax, and then calculates the mean.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()

        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias + torch.ones_like(conv.bias) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
