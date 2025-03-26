import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Dilation of the convolution.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int,
        padding: int,
        dilation: int,
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


# Constants
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 256
height = 256
stride = 1
padding = 0
dilation = 1
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
