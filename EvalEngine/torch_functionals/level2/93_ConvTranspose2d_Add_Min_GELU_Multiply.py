import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    add_value: float,
    multiply_value: float,
) -> torch.Tensor:
    """
    Applies transposed convolution, adds a value, takes minimum, applies GELU, and multiplies by a value.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        stride (int): Stride of the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        add_value (float): Value to add
        multiply_value (float): Value to multiply by

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose2d(x, conv_transpose, bias=conv_transpose_bias, stride=stride)
    x = x + add_value
    x = torch.min(x, torch.tensor(0.0))
    x = F.gelu(x)
    x = x * multiply_value
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.conv_transpose_parameter = conv.weight
        self.conv_transpose_bias = conv.bias
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x, stride, fn=module_fn):
        return fn(
            x,
            stride,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.add_value,
            self.multiply_value,
        )


batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width), stride]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
