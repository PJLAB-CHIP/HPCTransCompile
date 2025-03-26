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
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a 3D transposed convolution followed by bias addition and residual operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        output_padding (int): Additional size added to output shape
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor for addition

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
    original_x = x.clone().detach()
    x = x + bias
    x = x + original_x
    x = x * original_x
    x = x + original_x
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum,
    a residual add, a multiplication, and another residual add.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose_parameter = conv_transpose.weight
        self.conv_transpose_bias = nn.Parameter(
            conv_transpose.bias + torch.ones_like(conv_transpose.bias) * 0.02
        )  # make sure its nonzero
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, stride, padding, output_padding, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            output_padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.bias_parameter,
        )


batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)


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
        bias_shape,
    ]
