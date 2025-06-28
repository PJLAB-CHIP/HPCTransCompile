import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv1_weight: nn.Parameter,
    conv2_weight: nn.Parameter,
    bn1_weight: nn.Parameter,
    bn1_bias: nn.Parameter,
    bn1_mean: nn.Parameter,
    bn1_var: nn.Parameter,
    bn2_weight: nn.Parameter,
    bn2_bias: nn.Parameter,
    bn2_mean: nn.Parameter,
    bn2_var: nn.Parameter,
    downsample_conv_weight: nn.Parameter,
    downsample_bn_weight: nn.Parameter,
    downsample_bn_bias: nn.Parameter,
    downsample_bn_mean: nn.Parameter,
    downsample_bn_var: nn.Parameter,
    stride: int,
    is_training: bool,
) -> torch.Tensor:
    """
    Implements the ResNet BasicBlock module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        conv1_weight (nn.Parameter): Weight tensor for first conv layer
        conv2_weight (nn.Parameter): Weight tensor for second conv layer
        bn1_weight (nn.Parameter): Weight tensor for first batch norm
        bn1_bias (nn.Parameter): Bias tensor for first batch norm
        bn1_mean (nn.Parameter): Running mean tensor for first batch norm
        bn1_var (nn.Parameter): Running variance tensor for first batch norm
        bn2_weight (nn.Parameter): Weight tensor for second batch norm
        bn2_bias (nn.Parameter): Bias tensor for second batch norm
        bn2_mean (nn.Parameter): Running mean tensor for second batch norm
        bn2_var (nn.Parameter): Running variance tensor for second batch norm
        downsample_conv_weight (nn.Parameter): Weight tensor for downsample conv
        downsample_bn_weight (nn.Parameter): Weight tensor for downsample batch norm
        downsample_bn_bias (nn.Parameter): Bias tensor for downsample batch norm
        downsample_bn_mean (nn.Parameter): Running mean tensor for downsample batch norm
        downsample_bn_var (nn.Parameter): Running variance tensor for downsample batch norm
        stride (int): Stride for first conv and downsample
        is_training (bool): Whether in training mode

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, out_channels, height//stride, width//stride)
    """
    identity = x

    # First conv block
    out = F.conv2d(x, conv1_weight, bias=None, stride=stride, padding=1)
    out = F.batch_norm(out, bn1_mean, bn1_var, bn1_weight, bn1_bias, is_training)
    out = F.relu(out)

    # Second conv block
    out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
    out = F.batch_norm(out, bn2_mean, bn2_var, bn2_weight, bn2_bias, is_training)

    # Downsample path
    identity = F.conv2d(x, downsample_conv_weight, bias=None, stride=stride)
    identity = F.batch_norm(
        identity,
        downsample_bn_mean,
        downsample_bn_var,
        downsample_bn_weight,
        downsample_bn_bias,
        is_training,
    )

    # Add and final activation
    out += identity
    out = F.relu(out)

    return out


class Model(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        """
        super(Model, self).__init__()

        # Extract conv1 parameters
        conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv1_weight = nn.Parameter(conv1.weight.data.clone())

        # Extract bn1 parameters
        bn1 = nn.BatchNorm2d(out_channels)
        self.bn1_weight = nn.Parameter(bn1.weight.data.clone())
        self.bn1_bias = nn.Parameter(bn1.bias.data.clone())
        self.bn1_mean = nn.Parameter(bn1.running_mean.data.clone())
        self.bn1_var = nn.Parameter(bn1.running_var.data.clone())

        # Extract conv2 parameters
        conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2_weight = nn.Parameter(conv2.weight.data.clone())

        # Extract bn2 parameters
        bn2 = nn.BatchNorm2d(out_channels)
        self.bn2_weight = nn.Parameter(bn2.weight.data.clone())
        self.bn2_bias = nn.Parameter(bn2.bias.data.clone())
        self.bn2_mean = nn.Parameter(bn2.running_mean.data.clone())
        self.bn2_var = nn.Parameter(bn2.running_var.data.clone())

        # Extract downsample parameters
        downsample_conv = nn.Conv2d(
            in_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.downsample_conv_weight = nn.Parameter(downsample_conv.weight.data.clone())

        downsample_bn = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample_bn_weight = nn.Parameter(downsample_bn.weight.data.clone())
        self.downsample_bn_bias = nn.Parameter(downsample_bn.bias.data.clone())
        self.downsample_bn_mean = nn.Parameter(downsample_bn.running_mean.data.clone())
        self.downsample_bn_var = nn.Parameter(downsample_bn.running_var.data.clone())

        self.stride = stride

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv1_weight,
            self.conv2_weight,
            self.bn1_weight,
            self.bn1_bias,
            self.bn1_mean,
            self.bn1_var,
            self.bn2_weight,
            self.bn2_bias,
            self.bn2_mean,
            self.bn2_var,
            self.downsample_conv_weight,
            self.downsample_bn_weight,
            self.downsample_bn_bias,
            self.downsample_bn_mean,
            self.downsample_bn_var,
            self.stride,
            self.training,
        )


# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]


def get_init_inputs():
    return [in_channels, out_channels, stride]
