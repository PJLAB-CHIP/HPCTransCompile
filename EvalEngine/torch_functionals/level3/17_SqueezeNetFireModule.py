import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    squeeze_weight: torch.Tensor,
    squeeze_bias: torch.Tensor,
    expand1x1_weight: torch.Tensor,
    expand1x1_bias: torch.Tensor,
    expand3x3_weight: torch.Tensor,
    expand3x3_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Implements the SqueezeNet Fire Module.

    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_channels, height, width)
        squeeze_weight (torch.Tensor): Weight tensor for squeeze conv
        squeeze_bias (torch.Tensor): Bias tensor for squeeze conv
        expand1x1_weight (torch.Tensor): Weight tensor for 1x1 expand conv
        expand1x1_bias (torch.Tensor): Bias tensor for 1x1 expand conv
        expand3x3_weight (torch.Tensor): Weight tensor for 3x3 expand conv
        expand3x3_bias (torch.Tensor): Bias tensor for 3x3 expand conv

    Returns:
        torch.Tensor: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
    """
    x = F.conv2d(x, squeeze_weight, squeeze_bias)
    x = F.relu(x)

    x1 = F.conv2d(x, expand1x1_weight, expand1x1_bias)
    x1 = F.relu(x1)

    x3 = F.conv2d(x, expand3x3_weight, expand3x3_bias, padding=1)
    x3 = F.relu(x3)

    return torch.cat([x1, x3], 1)


class Model(nn.Module):
    def __init__(
        self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels
    ):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(Model, self).__init__()

        # Extract parameters from squeeze conv
        squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_weight = nn.Parameter(squeeze.weight.data.clone())
        self.squeeze_bias = nn.Parameter(squeeze.bias.data.clone())

        # Extract parameters from expand1x1 conv
        expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_weight = nn.Parameter(expand1x1.weight.data.clone())
        self.expand1x1_bias = nn.Parameter(expand1x1.bias.data.clone())

        # Extract parameters from expand3x3 conv
        expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )
        self.expand3x3_weight = nn.Parameter(expand3x3.weight.data.clone())
        self.expand3x3_bias = nn.Parameter(expand3x3.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.squeeze_weight,
            self.squeeze_bias,
            self.expand1x1_weight,
            self.expand1x1_bias,
            self.expand3x3_weight,
            self.expand3x3_bias,
        )


# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64


def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]


def get_init_inputs():
    return [
        num_input_features,
        squeeze_channels,
        expand1x1_channels,
        expand3x3_channels,
    ]
