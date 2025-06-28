import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution followed by ReLU, LeakyReLU, GELU, Sigmoid activations and bias addition.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): 3D convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Bias tensor for 3D convolution of shape (out_channels)
        bias (torch.Tensor): Bias tensor for addition of shape (out_channels, 1, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution and activations
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = F.relu(x)
    x = F.leaky_relu(x, negative_slope=0.01)
    x = F.gelu(x)
    x = torch.sigmoid(x)
    x = x + bias
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.bias = self.bias

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.bias)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
