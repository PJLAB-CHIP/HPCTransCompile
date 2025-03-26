import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a transposed 1D convolution operation.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        output_padding (int): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv_transpose1d(
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
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        output_padding (int): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose1d(
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
        self.output_padding = output_padding
        self.groups = groups

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


# Constants for default arguments
batch_size = 16
in_channels = 64
out_channels = 3
kernel_size = 3
length = 128
stride = 1
padding = 0
output_padding = 0
groups = 1
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
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
