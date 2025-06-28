import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    gemm_weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    batch_norm_weight: torch.Tensor,
    batch_norm_bias: torch.Tensor,
    batch_norm_running_mean: torch.Tensor,
    batch_norm_running_var: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Performs GEMM, BatchNorm, GELU, GroupNorm, Mean, and ReLU operations in sequence.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        gemm_weight (torch.Tensor): Weight matrix for linear layer
        gemm_bias (torch.Tensor): Bias vector for linear layer
        batch_norm_weight (torch.Tensor): BatchNorm scale parameter
        batch_norm_bias (torch.Tensor): BatchNorm bias parameter
        batch_norm_running_mean (torch.Tensor): BatchNorm running mean
        batch_norm_running_var (torch.Tensor): BatchNorm running variance
        group_norm_weight (torch.Tensor): GroupNorm scale parameter
        group_norm_bias (torch.Tensor): GroupNorm bias parameter
        num_groups (int): Number of groups for GroupNorm

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, gemm_weight, gemm_bias)
    x = F.batch_norm(
        x,
        batch_norm_running_mean,
        batch_norm_running_var,
        batch_norm_weight,
        batch_norm_bias,
        training=True,
    )
    x = F.gelu(x)
    x = F.group_norm(x, num_groups, group_norm_weight, group_norm_bias)
    x = torch.mean(x, dim=1, keepdim=True)
    x = F.relu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, GroupNorm, Mean, and ReLU operations in sequence.
    """

    def __init__(self, in_features, out_features, num_groups):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        bn = nn.BatchNorm1d(out_features)
        gn = nn.GroupNorm(num_groups, out_features)
        self.gemm_weight = nn.Parameter(gemm.weight)
        self.gemm_bias = nn.Parameter(gemm.bias + torch.randn_like(gemm.bias) * 0.02)

        self.batch_norm_weight = nn.Parameter(
            bn.weight + torch.randn_like(bn.weight) * 0.02
        )
        self.batch_norm_bias = nn.Parameter(bn.bias + torch.randn_like(bn.bias) * 0.02)
        self.register_buffer("batch_norm_running_mean", torch.randn(out_features))
        self.register_buffer("batch_norm_running_var", torch.randn(out_features).abs())

        self.group_norm_weight = nn.Parameter(
            gn.weight + torch.randn_like(gn.weight) * 0.02
        )
        self.group_norm_bias = nn.Parameter(gn.bias + torch.randn_like(gn.bias) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.gemm_weight,
            self.gemm_bias,
            self.batch_norm_weight,
            self.batch_norm_bias,
            self.batch_norm_running_mean,
            self.batch_norm_running_var,
            self.group_norm_weight,
            self.group_norm_bias,
            num_groups,
        )


batch_size = 128
in_features = 512
out_features = 1024
num_groups = 8


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups]
