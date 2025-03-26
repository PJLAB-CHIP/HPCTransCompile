import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_bias: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Applies GEMM, BiasAdd, Hardtanh, Mish and GroupNorm operations in sequence.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix for linear layer of shape (out_features, in_features)
        weight_bias (torch.Tensor): Bias tensor for linear layer of shape (out_features,)
        bias (torch.Tensor): Additional bias tensor of shape (out_features,)
        num_groups (int): Number of groups for group normalization
        eps (float): Small constant added for numerical stability in group norm

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, weight_bias)
    x = x + bias
    x = F.hardtanh(x)
    x = F.mish(x)
    x = F.group_norm(x, num_groups=num_groups, eps=eps)
    return x


class Model(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """

    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = gemm.weight
        self.weight_bias = gemm.bias
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.num_groups = num_groups

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.weight_bias, self.bias, self.num_groups)


batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]
