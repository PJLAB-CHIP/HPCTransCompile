import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Weight tensor of shape (in_channels, 1, kernel_size, kernel_size).
        bias (torch.Tensor): Bias tensor of shape (in_channels).
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        groups (int): Number of groups in the convolution.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
    """
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self, in_channels: int, kernel_size: int, stride: int, padding: int, bias: bool
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        # Copy the initialized parameters
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None
        self.stride = stride
        self.padding = padding
        self.groups = in_channels

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.groups,
        )


# Constants
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, bias]
