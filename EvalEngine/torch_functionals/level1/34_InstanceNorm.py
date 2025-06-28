import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Functional instance normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width)
        weight (torch.Tensor): Scale parameter
        bias (torch.Tensor): Shift parameter
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
    """
    N, C, H, W = x.size()

    # Calculate mean and variance along spatial dimensions
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)

    # Normalize
    x = (x - mean) / torch.sqrt(var + eps)

    # Apply affine transform
    if weight is not None and bias is not None:
        x = x * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)

    return x


class Model(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """

    def __init__(self, num_features: int, eps: float):
        """
        Initializes the InstanceNorm parameters.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias, self.eps)


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
