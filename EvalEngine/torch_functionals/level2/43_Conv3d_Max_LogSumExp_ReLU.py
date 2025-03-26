import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, max pooling, log sum exp, and ReLU activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the convolution
        padding (int): Padding of the convolution
        conv_weight (torch.Tensor): Convolution weight tensor
        conv_bias (torch.Tensor): Convolution bias tensor

    Returns:
        torch.Tensor: Output tensor after applying convolution, max pooling, logsumexp and ReLU
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias, stride=stride, padding=padding)
    x = F.max_pool3d(x, kernel_size=2, stride=2)
    x = torch.logsumexp(x, dim=1, keepdim=True)
    x = F.relu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(
            conv.bias
            + torch.randn(
                conv.bias.shape, device=conv.bias.device, dtype=conv.bias.dtype
            )
            * 0.02
        )

    def forward(self, x, stride, padding, fn=module_fn):
        return fn(x, stride, padding, self.conv_weight, self.conv_bias)


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width), stride, padding]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
