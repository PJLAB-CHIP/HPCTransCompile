import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    multiplier: float,
    negative_slope: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies linear transformation, multiplies by scalar, and applies LeakyReLU.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        multiplier (float): Scalar multiplier
        negative_slope (float): Negative slope for LeakyReLU
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = x * multiplier
    x = F.leaky_relu(x, negative_slope=negative_slope)
    return x


class Model(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = gemm.weight
        self.bias = gemm.bias

    def forward(self, x, fn=module_fn):
        return fn(x, multiplier, negative_slope, self.weight, self.bias)


batch_size = 128
in_features = 1024
out_features = 512
multiplier = 2.0
negative_slope = 0.1


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
