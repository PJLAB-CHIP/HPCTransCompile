import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a 3D transposed convolution, scaling, average pooling, bias addition and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        scale1 (torch.Tensor): First scaling factor
        scale2 (torch.Tensor): Second scaling factor
        bias (torch.Tensor): Bias tensor for addition

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = x * scale1
    x = F.avg_pool3d(x, kernel_size=2)
    x = x + bias
    x = x * scale2
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale1,
        scale2,
        bias_shape,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_transpose_parameter = nn.Parameter(conv_transpose.weight)
        self.conv_transpose_bias = nn.Parameter(
            conv_transpose.bias
            + torch.randn(
                conv_transpose.bias.shape,
                device=conv_transpose.bias.device,
                dtype=conv_transpose.bias.dtype,
            )
            * 0.02
        )
        self.scale1_parameter = nn.Parameter(torch.tensor(scale1))
        self.scale2_parameter = nn.Parameter(torch.tensor(scale2))
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, stride, padding, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.scale1_parameter,
            self.scale2_parameter,
            self.bias_parameter,
        )


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width), stride, padding]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale1,
        scale2,
        bias_shape,
    ]
