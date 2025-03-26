import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Functional implementation of a neural network layer that:
    1. Applies a 2D convolution with learnable weights and biases
    2. Applies ReLU activation function
    3. Adds a learnable bias term

    Args:
        x (Tensor): Input tensor of shape (N, C_in, H, W)
        conv_weight (Tensor): Convolution weights of shape (C_out, C_in, kernel_size, kernel_size)
        conv_bias (Tensor): Convolution bias of shape (C_out)
        bias (Tensor): Additional bias term of shape (C_out, 1, 1)

    Returns:
        Tensor: Output tensor of shape (N, C_out, H_out, W_out)
    """
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.relu(x)
    x = x + bias
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.bias)


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
