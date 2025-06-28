import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    scaling_factor: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    bn_eps: torch.Tensor,
    bn_momentum: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, batch normalization and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        scaling_factor (float): Factor to scale the output by
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        bn_weight (torch.Tensor): BatchNorm weight (gamma)
        bn_bias (torch.Tensor): BatchNorm bias (beta)
        bn_running_mean (torch.Tensor): BatchNorm running mean
        bn_running_var (torch.Tensor): BatchNorm running variance

    Returns:
        torch.Tensor: Output tensor after convolution, batch norm and scaling
    """
    x = F.conv2d(x, conv_weight, conv_bias)
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
    x = x * scaling_factor
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        bn = nn.BatchNorm2d(out_channels)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)

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

        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.scaling_factor,
            self.conv_weight,
            self.conv_bias,
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
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
