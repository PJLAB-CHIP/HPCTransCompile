import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    scaling_factor: float,
    pool_kernel_size: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, tanh activation, scaling, bias addition and max pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        scaling_factor (float): Factor to scale the tensor by after tanh
        pool_kernel_size (int): Size of max pooling kernel
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        bias (torch.Tensor): Bias tensor for addition of shape (out_channels, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution, tanh, scaling, bias and max pooling
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = torch.tanh(x)
    x = x * scaling_factor
    x = x + bias
    x = F.max_pool2d(x, pool_kernel_size)
    return x


class Model(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        scaling_factor,
        bias_shape,
        pool_kernel_size,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, scaling_factor, pool_kernel_size, fn=module_fn):
        return fn(
            x,
            scaling_factor,
            pool_kernel_size,
            self.conv_weight,
            self.conv_bias,
            self.bias,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, height, width),
        scaling_factor,
        pool_kernel_size,
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        scaling_factor,
        bias_shape,
        pool_kernel_size,
    ]
