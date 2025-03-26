import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies transposed convolution, global average pooling, bias addition, log-sum-exp, sum and multiplication.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor for addition

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose2d(x, conv_transpose, bias=conv_transpose_bias)
    x = torch.mean(x, dim=(2, 3), keepdim=True)
    x = x + bias
    x = torch.logsumexp(x, dim=1, keepdim=True)
    x = torch.sum(x, dim=(2, 3))
    x = x * 10.0
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias,
    applies log-sum-exp, sum, and multiplication.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(
            conv.bias
            + torch.randn(
                conv.bias.shape, device=conv.bias.device, dtype=conv.bias.dtype
            )
            * 0.02
        )
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.bias_parameter,
        )


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
