import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    gemm_weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    multiply_weight: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Performs GEMM, GroupNorm, Swish, Multiply, and Swish operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        gemm_weight (torch.Tensor): Weight matrix for linear layer of shape (out_features, in_features)
        gemm_bias (torch.Tensor): Bias vector for linear layer of shape (out_features)
        group_norm_weight (torch.Tensor): Weight parameter for group norm of shape (out_features)
        group_norm_bias (torch.Tensor): Bias parameter for group norm of shape (out_features)
        multiply_weight (torch.Tensor): Weight tensor for multiplication of shape (out_features)
        num_groups (int): Number of groups for group normalization

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, gemm_weight, gemm_bias)
    x = F.group_norm(x, num_groups, group_norm_weight, group_norm_bias)
    x = x * torch.sigmoid(x)
    x = x * multiply_weight
    x = x * torch.sigmoid(x)
    return x


class Model(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """

    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.gemm_weight = gemm.weight
        self.gemm_bias = gemm.bias
        group_norm = nn.GroupNorm(num_groups, out_features)
        self.group_norm_weight = group_norm.weight
        self.group_norm_bias = group_norm.bias
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape) * 0.02)
        self.num_groups = num_groups

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.gemm_weight,
            self.gemm_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.multiply_weight,
            self.num_groups,
        )


batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
