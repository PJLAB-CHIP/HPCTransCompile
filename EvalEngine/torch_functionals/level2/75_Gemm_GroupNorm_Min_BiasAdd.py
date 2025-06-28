import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    gemm_weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs GEMM, Group Normalization, Minimum operation and Bias addition.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        gemm_weight (torch.Tensor): Weight matrix for linear layer of shape (out_features, in_features)
        gemm_bias (torch.Tensor): Bias vector for linear layer of shape (out_features)
        group_norm_weight (torch.Tensor): Weight parameter for group norm of shape (out_features)
        group_norm_bias (torch.Tensor): Bias parameter for group norm of shape (out_features)
        num_groups (int): Number of groups for group normalization
        bias (torch.Tensor): Bias tensor for final addition of shape (1, out_features, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying GEMM, group norm, min and bias addition
    """
    x = F.linear(x, gemm_weight, gemm_bias)
    # Reshape for group norm
    x = F.group_norm(x, num_groups, group_norm_weight, group_norm_bias)
    x = torch.min(x, dim=1, keepdim=True)[0]
    x = x + bias
    return x


class Model(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.gemm_weight = nn.Parameter(gemm.weight)
        self.gemm_bias = nn.Parameter(gemm.bias)
        gn = nn.GroupNorm(num_groups, out_features)
        self.group_norm_weight = nn.Parameter(
            gn.weight + torch.randn(gn.weight.shape) * 0.02
        )
        self.group_norm_bias = nn.Parameter(gn.bias + torch.randn(gn.bias.shape) * 0.02)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.num_groups = num_groups

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.gemm_weight,
            self.gemm_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.num_groups,
            self.bias,
        )


batch_size = 128
in_features = 512
out_features = 256
num_groups = 8
bias_shape = (1, out_features, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
