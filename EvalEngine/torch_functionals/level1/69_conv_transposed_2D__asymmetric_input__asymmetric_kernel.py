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
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        output_padding (tuple): Additional size added to one side of the output shape.
        dilation (tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv_transpose2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple): Tuple of integers representing the stride of the convolution.
        padding (tuple): Tuple of integers representing the padding applied to the input.
        output_padding (tuple): Tuple of integers representing the additional size added to one side of the output shape.
        dilation (tuple): Tuple of integers representing the spacing between kernel elements.
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
        dilation,
        groups,
        bias,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
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
        self.dilation = dilation

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )


# Constants for default arguments
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height_in = 16
width_in = 32
stride = (1, 1)
padding = (0, 0)
output_padding = (0, 0)
dilation = (1, 1)
groups = 1
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        bias,
    ]
