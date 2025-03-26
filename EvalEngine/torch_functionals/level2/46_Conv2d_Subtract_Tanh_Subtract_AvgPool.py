import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    kernel_size_pool: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    subtract1_value: float,
    subtract2_value: float,
) -> torch.Tensor:
    """
    Applies convolution, subtraction, tanh activation, subtraction and average pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        kernel_size_pool (int): Kernel size for average pooling
        conv_weight (torch.Tensor): Convolution weight tensor
        conv_bias (torch.Tensor): Convolution bias tensor
        subtract1_value (float): First subtraction value
        subtract2_value (float): Second subtraction value

    Returns:
        torch.Tensor: Output tensor after applying convolution, subtractions, tanh and avg pooling
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x - subtract1_value
    x = torch.tanh(x)
    x = x - subtract2_value
    x = F.avg_pool2d(x, kernel_size_pool)
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        subtract1_value,
        subtract2_value,
        kernel_size_pool,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(
            conv.bias
            + torch.randn(
                conv.bias.shape, device=conv.bias.device, dtype=conv.bias.dtype
            )
            * 0.02
        )
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.kernel_size_pool,
            self.conv_weight,
            self.conv_bias,
            self.subtract1_value,
            self.subtract2_value,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        subtract1_value,
        subtract2_value,
        kernel_size_pool,
    ]
