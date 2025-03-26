import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
) -> torch.Tensor:
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

    Args:
        x (torch.Tensor): Input tensor
        stride (tuple): Stride of convolution
        padding (tuple): Padding to apply
        weight (torch.Tensor): Convolution weights
        bias (torch.Tensor): Bias tensor (optional)

    Returns:
        torch.Tensor: Output tensor
    """
    return F.conv_transpose2d(x, weight, bias=bias, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple): Stride of the convolution (height, width).
        padding (tuple): Padding applied to the input (height, width).
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        bias: bool,
    ):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Copy the initialized parameters
        self.weight = nn.Parameter(self.conv_transpose2d.weight.clone())
        self.bias = nn.Parameter(self.conv_transpose2d.bias.clone()) if bias else None

        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            fn: Function to use for forward pass

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
        )


# Constants
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (1, 1)
padding = (1, 2)
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias]
