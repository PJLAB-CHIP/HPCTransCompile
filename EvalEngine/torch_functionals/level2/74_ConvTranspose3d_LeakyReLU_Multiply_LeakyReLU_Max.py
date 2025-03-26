import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, LeakyReLU, multiplication, LeakyReLU and max pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride for the transposed convolution
        padding (int): Padding for the transposed convolution
        output_padding (int): Output padding for the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        multiplier (torch.Tensor): Multiplier tensor of shape (out_channels, 1, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = F.leaky_relu(x, negative_slope=0.2)
    x = x * multiplier
    x = F.leaky_relu(x, negative_slope=0.2)
    x = F.max_pool3d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter,
    applies LeakyReLU again, and performs a max pooling operation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        multiplier_shape,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)
        self.multiplier_parameter = nn.Parameter(torch.randn(multiplier_shape) * 0.02)

    def forward(self, x, stride, padding, output_padding, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            output_padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.multiplier_parameter,
        )


batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, depth, height, width),
        stride,
        padding,
        output_padding,
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        multiplier_shape,
    ]
