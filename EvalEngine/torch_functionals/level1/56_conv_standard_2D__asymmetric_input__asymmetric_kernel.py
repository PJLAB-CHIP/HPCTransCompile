import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Implementation of 2D convolution with asymmetric kernel.

    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width).
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size[0], kernel_size[1]).
        bias: Bias tensor of shape (out_channels).
        stride: Stride of the convolution.
        padding: Padding of the convolution.
        dilation: Dilation of the convolution.
        groups: Number of groups in the convolution.

    Returns:
        Output tensor of shape (batch_size, out_channels, height, width).
    """
    return F.conv2d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        groups: int,
        bias: bool,
    ):
        super(Model, self).__init__()
        # Create a Conv2d layer to get the same initialization
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # Copy the initialized parameters
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(
        self,
        x: torch.Tensor,
        fn=module_fn,
    ) -> torch.Tensor:
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Constants for default arguments
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
height = 256
width = 128  # Asymmetric input dimensions
stride = (1, 1)
padding = (0, 0)
dilation = (1, 1)
groups = 1
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    ]
