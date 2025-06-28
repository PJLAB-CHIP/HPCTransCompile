import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    eps: float,
    momentum: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, activation, and batch normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        eps (float): Small constant for numerical stability in batch norm
        momentum (float): Momentum for batch norm running stats
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        bn_weight (torch.Tensor): Batch norm weight (gamma)
        bn_bias (torch.Tensor): Batch norm bias (beta)
        bn_running_mean (torch.Tensor): Batch norm running mean
        bn_running_var (torch.Tensor): Batch norm running variance

    Returns:
        torch.Tensor: Output after convolution, activation and batch norm
    """
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.multiply(torch.tanh(F.softplus(x)), x)
    x = F.batch_norm(
        x,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=True,
        momentum=momentum,
        eps=eps,
    )
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size, eps, momentum):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias + torch.randn(conv.bias.shape) * 0.02)
        self.bn_weight = nn.Parameter(bn.weight)
        self.bn_bias = nn.Parameter(bn.bias + torch.randn(bn.bias.shape) * 0.02)
        self.register_buffer(
            "bn_running_mean",
            bn.running_mean + torch.randn(bn.running_mean.shape) * 0.02,
        )
        self.register_buffer(
            "bn_running_var",
            bn.running_var + torch.randn(bn.running_var.shape).abs() * 0.02,
        )

    def forward(self, x, eps, momentum, fn=module_fn):
        return fn(
            x,
            eps,
            momentum,
            self.conv_weight,
            self.conv_bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
        )


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
eps = 1e-5
momentum = 0.1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width), eps, momentum]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, eps, momentum]
