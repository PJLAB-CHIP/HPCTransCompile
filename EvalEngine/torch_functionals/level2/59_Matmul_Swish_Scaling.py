import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """
    Applies linear transformation, Swish activation, and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)
        scaling_factor (float): Factor to scale the output by

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = x * torch.sigmoid(x)  # Swish activation
    x = x * scaling_factor
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """

    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias, self.scaling_factor)


batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
