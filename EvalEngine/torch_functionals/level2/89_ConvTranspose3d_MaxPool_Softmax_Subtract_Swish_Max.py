import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    pool_kernel_size: int,
    pool_stride: int,
    pool_padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    subtract: torch.Tensor,
) -> torch.Tensor:
    """
    Applies sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride for conv transpose
        padding (int): Padding for conv transpose
        output_padding (int): Output padding for conv transpose
        pool_kernel_size (int): Kernel size for max pooling
        pool_stride (int): Stride for max pooling
        pool_padding (int): Padding for max pooling
        conv_transpose (torch.Tensor): Weight tensor for transposed convolution
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        subtract (torch.Tensor): Subtraction parameter tensor
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = F.max_pool3d(
        x, kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
    )
    x = F.softmax(x, dim=1)
    x = x - subtract.view(1, -1, 1, 1, 1)
    x = torch.sigmoid(x) * x  # Swish
    x = torch.max(x, dim=1)[0]
    return x


class Model(nn.Module):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_transpose_parameter = conv_transpose.weight
        self.conv_transpose_bias = conv_transpose.bias
        self.subtract_parameter = nn.Parameter(torch.randn(out_channels) * 0.02)

    def forward(
        self,
        x,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        fn=module_fn,
    ):
        return fn(
            x,
            stride,
            padding,
            output_padding,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.subtract_parameter,
        )


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, depth, height, width),
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding,
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding,
    ]
