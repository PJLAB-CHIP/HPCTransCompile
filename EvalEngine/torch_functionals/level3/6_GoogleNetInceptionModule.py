import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    branch1x1_weight: nn.Parameter,
    branch1x1_bias: nn.Parameter,
    branch3x3_reduce_weight: nn.Parameter,
    branch3x3_reduce_bias: nn.Parameter,
    branch3x3_weight: nn.Parameter,
    branch3x3_bias: nn.Parameter,
    branch5x5_reduce_weight: nn.Parameter,
    branch5x5_reduce_bias: nn.Parameter,
    branch5x5_weight: nn.Parameter,
    branch5x5_bias: nn.Parameter,
    branch_pool_conv_weight: nn.Parameter,
    branch_pool_conv_bias: nn.Parameter,
) -> torch.Tensor:
    """
    Implements the GoogleNet Inception module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        branch*_weight (nn.Parameter): Weight tensors for each convolution
        branch*_bias (nn.Parameter): Bias tensors for each convolution

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, out_channels, height, width)
    """
    # 1x1 branch
    branch1x1 = F.conv2d(x, branch1x1_weight, branch1x1_bias)

    # 3x3 branch
    branch3x3 = F.conv2d(x, branch3x3_reduce_weight, branch3x3_reduce_bias)
    branch3x3 = F.conv2d(branch3x3, branch3x3_weight, branch3x3_bias, padding=1)

    # 5x5 branch
    branch5x5 = F.conv2d(x, branch5x5_reduce_weight, branch5x5_reduce_bias)
    branch5x5 = F.conv2d(branch5x5, branch5x5_weight, branch5x5_bias, padding=2)

    # Pool branch
    branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = F.conv2d(branch_pool, branch_pool_conv_weight, branch_pool_conv_bias)

    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
    return torch.cat(outputs, 1)


class Model(nn.Module):
    def __init__(
        self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj
    ):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(Model, self).__init__()

        # 1x1 branch parameters
        conv1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch1x1_weight = nn.Parameter(conv1x1.weight.data.clone())
        self.branch1x1_bias = nn.Parameter(conv1x1.bias.data.clone())

        # 3x3 branch parameters
        conv3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_reduce_weight = nn.Parameter(conv3x3_reduce.weight.data.clone())
        self.branch3x3_reduce_bias = nn.Parameter(conv3x3_reduce.bias.data.clone())

        conv3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        self.branch3x3_weight = nn.Parameter(conv3x3.weight.data.clone())
        self.branch3x3_bias = nn.Parameter(conv3x3.bias.data.clone())

        # 5x5 branch parameters
        conv5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_reduce_weight = nn.Parameter(conv5x5_reduce.weight.data.clone())
        self.branch5x5_reduce_bias = nn.Parameter(conv5x5_reduce.bias.data.clone())

        conv5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        self.branch5x5_weight = nn.Parameter(conv5x5.weight.data.clone())
        self.branch5x5_bias = nn.Parameter(conv5x5.bias.data.clone())

        # Pool branch parameters
        conv_pool = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        self.branch_pool_conv_weight = nn.Parameter(conv_pool.weight.data.clone())
        self.branch_pool_conv_bias = nn.Parameter(conv_pool.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.branch1x1_weight,
            self.branch1x1_bias,
            self.branch3x3_reduce_weight,
            self.branch3x3_reduce_bias,
            self.branch3x3_weight,
            self.branch3x3_bias,
            self.branch5x5_reduce_weight,
            self.branch5x5_reduce_bias,
            self.branch5x5_weight,
            self.branch5x5_bias,
            self.branch_pool_conv_weight,
            self.branch_pool_conv_bias,
        )


# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]
