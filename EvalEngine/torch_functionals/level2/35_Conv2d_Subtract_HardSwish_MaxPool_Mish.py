import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    subtract_value: float,
    pool_kernel_size: int,
) -> torch.Tensor:
    """
    Applies convolution, subtraction, HardSwish, MaxPool and Mish activations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        subtract_value (float): Value to subtract
        pool_kernel_size (int): Kernel size for max pooling

    Returns:
        torch.Tensor: Output tensor after applying convolution, subtraction,
            HardSwish, MaxPool and Mish activations
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x - subtract_value
    x = F.hardswish(x)
    x = F.max_pool2d(x, pool_kernel_size)
    x = F.mish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = nn.Parameter(
            conv.bias
            + torch.randn(
                conv.bias.shape, device=conv.bias.device, dtype=conv.bias.dtype
            )
            * 0.02
        )
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.subtract_value,
            self.pool_kernel_size,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
