import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    Applies a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling
    3. Addition
    4. Softmax
    5. Tanh activation
    6. Scaling

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor for addition
        scaling_factor (float): Scaling factor for final multiplication
        stride (int): Stride for transposed convolution
        padding (int): Padding for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = torch.mean(x, dim=1, keepdim=True)
    x = x + bias
    x = F.softmax(x, dim=1)
    x = torch.tanh(x)
    x = x * scaling_factor
    return x


class Model(nn.Module):
    """
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling
    3. Addition
    4. Softmax
    5. Tanh activation
    6. Scaling
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias_shape,
        scaling_factor,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

        self.conv_transpose_weight = conv_transpose.weight
        self.conv_transpose_bias = conv_transpose.bias
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.bias,
            self.scaling_factor,
            stride,
            padding,
        )


batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias_shape,
        scaling_factor,
    ]
