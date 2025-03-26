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
    bn_eps: torch.Tensor,
    bn_momentum: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a 3D transposed convolution, batch normalization and two average pooling layers.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bn_weight (torch.Tensor): Batch norm weight parameter
        bn_bias (torch.Tensor): Batch norm bias parameter
        bn_running_mean (torch.Tensor): Batch norm running mean
        bn_running_var (torch.Tensor): Batch norm running variance
        bn_eps (torch.Tensor): Small constant for numerical stability
        bn_momentum (torch.Tensor): Momentum for running stats

    Returns:
        torch.Tensor: Output tensor after applying transposed conv, batch norm and avg pooling
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
        momentum=bn_momentum,
        eps=bn_eps,
    )
    x = F.avg_pool3d(x, kernel_size=2)
    x = F.avg_pool3d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization,
    two average pooling layers.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias_shape
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        bn = nn.BatchNorm3d(out_channels)
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)

        self.bn_weight = nn.Parameter(bn.weight + torch.randn(bn.weight.shape) * 0.02)
        self.bn_bias = nn.Parameter(bn.bias + torch.randn(bn.bias.shape) * 0.02)
        self.register_buffer(
            "bn_running_mean",
            bn.running_mean + torch.randn(bn.running_mean.shape) * 0.02,
        )
        self.register_buffer(
            "bn_running_var",
            bn.running_var + torch.randn(bn.running_var.shape).abs() * 0.02,
        )
        self.register_buffer("bn_eps", torch.tensor(1e-5))
        self.register_buffer("bn_momentum", torch.tensor(0.1))

    def forward(self, x, stride, padding, fn=module_fn):
        return fn(
            x,
            stride,
            padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.bn_eps,
            self.bn_momentum,
        )


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width), stride, padding]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
