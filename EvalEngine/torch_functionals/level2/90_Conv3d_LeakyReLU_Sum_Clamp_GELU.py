import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    sum_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, LeakyReLU, tensor addition, clamping and GELU activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): 3D convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Bias tensor for 3D convolution of shape (out_channels)
        sum_tensor (torch.Tensor): Tensor to add of shape (out_channels, 1, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution, LeakyReLU, addition,
            clamping and GELU activation
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = F.leaky_relu(x, negative_slope=0.2)
    x = x + sum_tensor
    x = torch.clamp(x, min=-1.0, max=1.0)
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.sum_tensor)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]
