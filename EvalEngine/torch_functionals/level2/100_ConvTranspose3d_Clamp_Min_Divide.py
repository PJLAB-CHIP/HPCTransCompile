import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    min_value: float,
    divisor: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a transposed 3D convolution, clamps output to min value, and divides by constant.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        min_value (float): Minimum value for clamping
        divisor (float): Value to divide output by
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying transposed convolution, clamping and division
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = torch.clamp(x, min=min_value)
    x = x / divisor
    return x


class Model(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value,
    and then divides the result by a constant.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.conv_transpose_parameter = conv_transpose.weight
        self.conv_transpose_bias = conv_transpose.bias

    def forward(self, x, stride, padding, min_value, divisor, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            min_value,
            divisor,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
        )


batch_size = 16
in_channels = 32
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, depth, height, width),
        stride,
        padding,
        min_value,
        divisor,
    ]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
