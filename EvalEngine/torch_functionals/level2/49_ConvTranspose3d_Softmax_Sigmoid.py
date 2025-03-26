import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    bias_flag: bool,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a 3D transposed convolution operation followed by softmax and sigmoid.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        output_padding (int): Additional size added to output shape
        bias_flag (bool): Whether to use bias in conv_transpose
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying transposed convolution, softmax and sigmoid
    """
    bias = conv_transpose_bias if bias_flag else None
    x = F.conv_transpose3d(
        x,
        conv_transpose,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = F.softmax(x, dim=1)
    x = torch.sigmoid(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.conv_transpose_parameter = nn.Parameter(conv_transpose.weight)
        self.conv_transpose_bias = (
            nn.Parameter(
                conv_transpose.bias
                + torch.randn(
                    conv_transpose.bias.shape,
                    device=conv_transpose.bias.device,
                    dtype=conv_transpose.bias.dtype,
                )
                * 0.02
            )
            if bias
            else None
        )

    def forward(self, x, stride, padding, output_padding, bias, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            output_padding,
            bias,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
        )


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = True


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, D, H, W),
        stride,
        padding,
        output_padding,
        bias,
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias,
    ]
