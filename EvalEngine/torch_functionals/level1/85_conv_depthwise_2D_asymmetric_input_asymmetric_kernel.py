import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
        weight (torch.Tensor): Weight tensor of shape (in_channels, out_channels//in_channels, kernel_size_h, kernel_size_w).
        bias (torch.Tensor): Bias tensor of shape (out_channels).
        stride_h (int): Stride of the convolution in height dimension.
        stride_w (int): Stride of the convolution in width dimension.
        padding_h (int): Padding applied to the input in height dimension.
        padding_w (int): Padding applied to the input in width dimension.
        dilation_h (int): Spacing between kernel elements in height dimension.
        dilation_w (int): Spacing between kernel elements in width dimension.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(
        x,
        weight,
        bias=bias,
        stride=(stride_h, stride_w),
        padding=(padding_h, padding_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        bias,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=groups,
        )
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )


# Constants
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        bias,
    ]
