import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        eps (float): Small value added to denominator for numerical stability

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied
    """
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms


class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """

    def __init__(self, num_features: int, eps: float):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor
            eps (float): Small value added to denominator for numerical stability
        """
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Forward pass that calls module_fn.

        Args:
            x (torch.Tensor): Input tensor
            fn: Function to call, defaults to module_fn

        Returns:
            torch.Tensor: Output of module_fn
        """
        return fn(x, self.eps)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256
eps = 1e-5


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features, eps]
