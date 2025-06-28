import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    scaling_factor: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, scaling, tanh, bias multiplication and sigmoid.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): 3D convolution weight tensor
        conv_bias (torch.Tensor): 3D convolution bias tensor
        scaling_factor (torch.Tensor): Scaling factor tensor of shape (out_channels, 1, 1, 1)
        bias (torch.Tensor): Bias tensor of shape (out_channels, 1, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution, scaling, tanh, bias and sigmoid
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = x * scaling_factor
    x = torch.tanh(x)
    x = x * bias
    x = torch.sigmoid(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(
            conv.bias
            + torch.randn(
                conv.bias.shape, device=conv.bias.device, dtype=conv.bias.dtype
            )
            * 0.02
        )
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.scaling_factor, self.bias)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
