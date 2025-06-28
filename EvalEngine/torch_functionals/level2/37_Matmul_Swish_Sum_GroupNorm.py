import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_bias: torch.Tensor,
    bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Performs matrix multiplication, Swish activation, bias addition and group normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        weight_bias (torch.Tensor): Bias vector of shape (out_features,)
        bias (torch.Tensor): Bias term of shape (out_features,)
        group_norm_weight (torch.Tensor): GroupNorm weight of shape (out_features,)
        group_norm_bias (torch.Tensor): GroupNorm bias of shape (out_features,)
        num_groups (int): Number of groups for GroupNorm

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, weight_bias)
    x = torch.sigmoid(x) * x  # Swish activation
    x = x + bias
    x = F.group_norm(x, num_groups, group_norm_weight, group_norm_bias)
    return x


class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        mm = nn.Linear(in_features, out_features)
        self.weight = mm.weight
        self.weight_bias = nn.Parameter(
            mm.bias
            + torch.randn(mm.bias.shape, device=mm.bias.device, dtype=mm.bias.dtype)
            * 0.02
        )
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        group_norm = nn.GroupNorm(num_groups, out_features)
        self.group_norm_weight = nn.Parameter(
            group_norm.weight
            + torch.randn(
                group_norm.weight.shape,
                device=group_norm.weight.device,
                dtype=group_norm.weight.dtype,
            )
            * 0.02
        )
        self.group_norm_bias = nn.Parameter(
            group_norm.bias
            + torch.randn(
                group_norm.bias.shape,
                device=group_norm.bias.device,
                dtype=group_norm.bias.dtype,
            )
            * 0.02
        )
        self.num_groups = num_groups

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.weight,
            self.weight_bias,
            self.bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.num_groups,
        )


batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
