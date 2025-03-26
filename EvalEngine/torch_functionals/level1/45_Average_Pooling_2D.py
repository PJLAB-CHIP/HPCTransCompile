import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    Applies 2D Average Pooling using functional interface.

    Args:
        x (torch.Tensor): Input tensor
        kernel_size (int): Size of pooling window
        stride (int): Stride of pooling operation
        padding (int): Input padding

    Returns:
        torch.Tensor: Output tensor with 2D Average Pooling applied
    """
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window
            stride (int): Stride of the pooling operation
            padding (int): Padding applied to input tensor
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            fn: Function to apply pooling operation, defaults to module_fn

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )


batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3
stride = None  # Defaults to kernel_size
padding = 0


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, stride if stride is not None else kernel_size, padding]
