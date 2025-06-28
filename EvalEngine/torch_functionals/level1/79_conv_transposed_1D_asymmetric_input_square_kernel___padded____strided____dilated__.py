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
) -> torch.Tensor:
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel. Supports padding, striding, and dilation.

    Args:
        x (torch.Tensor): Input tensor
        weight (torch.Tensor): Convolution weights
        bias (torch.Tensor): Bias tensor (optional)
        stride (int): Stride of the convolution
        padding (int): Padding applied to the input
        dilation (int): Spacing between kernel elements

    Returns:
        torch.Tensor: Output tensor
    """
    return F.conv_transpose1d(
        x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
    )


class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
    ):
        super(Model, self).__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # Copy the initialized parameters
        self.weight = nn.Parameter(self.conv_transpose1d.weight.clone())
        self.bias = nn.Parameter(self.conv_transpose1d.bias.clone()) if bias else None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )


# Constants
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 128
stride = 2
padding = 1
dilation = 2
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, bias]
