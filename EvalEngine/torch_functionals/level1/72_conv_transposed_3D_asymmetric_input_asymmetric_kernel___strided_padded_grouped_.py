import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (tuple): Stride for the convolution.
        padding (tuple): Padding for the convolution.
        output_padding (tuple): Output padding for the convolution.
        groups (int): Number of groups in the convolution.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv_transpose3d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        output_padding: tuple,
        groups: int,
        bias: bool,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

        # Copy the initialized parameters
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.output_padding = output_padding

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


# Constants
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 32
width = 64
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias,
    ]
