import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    scaling_factor: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Applies transposed convolution, bias addition, clamping, scaling, clamping and division.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        stride (int): Stride of the convolution
        padding (int): Zero-padding added to both sides of input
        output_padding (int): Additional size added to output shape
        scaling_factor (float): Factor to scale the tensor by
        conv_transpose (torch.Tensor): Transposed convolution weights
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor to add after convolution

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose2d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = x + bias
    x = torch.clamp(x, min=0.0, max=1.0)
    x = x * scaling_factor
    x = torch.clamp(x, min=0.0, max=1.0)
    x = x / scaling_factor
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
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
        scaling_factor,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose_parameter = nn.Parameter(conv_transpose.weight)
        self.conv_tranpose_bias = nn.Parameter(conv_transpose.bias)
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, stride, padding, output_padding, scaling_factor, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            output_padding,
            scaling_factor,
            self.conv_transpose_parameter,
            self.conv_tranpose_bias,
            self.bias_parameter,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, height, width),
        stride,
        padding,
        output_padding,
        scaling_factor,
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
        scaling_factor,
    ]
