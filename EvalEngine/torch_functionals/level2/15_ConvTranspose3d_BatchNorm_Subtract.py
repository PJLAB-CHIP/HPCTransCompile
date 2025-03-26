import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, batch norm, and mean subtraction.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        conv_transpose (torch.Tensor): Transposed conv weights
        conv_transpose_bias (torch.Tensor): Transposed conv bias
        bn_weight (torch.Tensor): BatchNorm weight parameter
        bn_bias (torch.Tensor): BatchNorm bias parameter
        bn_running_mean (torch.Tensor): BatchNorm running mean
        bn_running_var (torch.Tensor): BatchNorm running variance

    Returns:
        torch.Tensor: Output after conv transpose, batch norm and mean subtraction
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = F.batch_norm(
        x,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
    )
    x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
    return x


class Model(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        batch_norm = nn.BatchNorm3d(out_channels)

        self.conv_transpose_weight = conv_transpose.weight
        self.conv_transpose_bias = conv_transpose.bias
        self.bn_weight = batch_norm.weight
        self.bn_bias = batch_norm.bias
        self.register_buffer("bn_running_mean", batch_norm.running_mean)
        self.register_buffer("bn_running_var", batch_norm.running_var)

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
        )


batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias = True


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias]
