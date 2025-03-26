import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    scale: float,
    maxpool_kernel_size: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a transposed 3D convolution, scales the output, applies max pooling,
    global average pooling, and clamps the result.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        scale (float): Scaling factor to multiply output by
        maxpool_kernel_size (int): Kernel size for max pooling operation
        conv_transpose (torch.Tensor): Weight tensor for transposed convolution
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying all operations, with shape
            (batch_size, out_channels, 1, 1, 1)
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = x * scale
    x = F.max_pool3d(x, kernel_size=maxpool_kernel_size)
    x = F.adaptive_avg_pool3d(x, (1, 1, 1))
    x = torch.clamp(x, min=0, max=1)
    return x


class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling,
    global average pooling, and clamps the output.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale,
        maxpool_kernel_size,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_transpose_parameter = conv.weight
        self.conv_transpose_bias = conv.bias

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            scale,
            maxpool_kernel_size,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
        )


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale,
        maxpool_kernel_size,
    ]
