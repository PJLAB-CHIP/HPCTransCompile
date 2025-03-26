import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose_weight: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    sum_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    pool_kernel_size: tuple,
    norm_shape: tuple,
) -> torch.Tensor:
    """
    Functional implementation of a sequence of operations:
    1. 3D transposed convolution
    2. Addition with a learnable weight
    3. Layer normalization
    4. 3D average pooling
    5. GELU activation

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_transpose_weight (torch.Tensor): Weight tensor for transposed convolution
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        sum_weight (torch.Tensor): Learnable weight for addition
        norm_weight (torch.Tensor): Weight tensor for layer normalization
        norm_bias (torch.Tensor): Bias tensor for layer normalization
        stride (tuple): Stride for transposed convolution, as (depth_stride, height_stride, width_stride)
        padding (tuple): Padding for transposed convolution, as (depth_pad, height_pad, width_pad)
        output_padding (tuple): Output padding for transposed convolution, as (depth_pad, height_pad, width_pad)
        pool_kernel_size (tuple): Kernel size for average pooling, as (depth_kernel, height_kernel, width_kernel)
        norm_shape (tuple): Shape for layer normalization

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose_weight,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = x + sum_weight
    x = F.layer_norm(x, norm_shape, norm_weight, norm_bias)
    x = F.avg_pool3d(x, kernel_size=pool_kernel_size)
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        sum_weight,
        norm_shape,
        pool_kernel_size,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose_weight = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        norm = nn.LayerNorm(norm_shape)
        self.norm_weight = nn.Parameter(norm.weight + torch.randn(norm_shape) * 0.02)
        self.norm_bias = nn.Parameter(norm.bias + torch.randn(norm_shape) * 0.02)

    def forward(
        self,
        x,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        norm_shape,
        fn=module_fn,
    ):
        return fn(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.sum_weight,
            self.norm_weight,
            self.norm_bias,
            stride,
            padding,
            output_padding,
            pool_kernel_size,
            norm_shape,
        )


batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)


def get_inputs():
    return [
        torch.randn(batch_size, in_channels, depth, height, width),
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        norm_shape,
    ]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        sum_weight,
        norm_shape,
        pool_kernel_size,
    ]
