import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Applies Max Pooling 2D using functional interface.

    Args:
        x (torch.Tensor): Input tensor
        kernel_size (int): Size of pooling window
        stride (int): Stride of pooling window
        padding (int): Padding to be applied
        dilation (int): Spacing between kernel elements

    Returns:
        torch.Tensor: Output tensor after max pooling
    """
    return F.max_pool2d(
        x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )


class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the model parameters.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Forward pass that calls module_fn.
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )


batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
