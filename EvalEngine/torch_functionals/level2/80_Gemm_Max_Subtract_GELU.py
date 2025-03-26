import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    max_dim: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs a GEMM, followed by a max operation, subtraction, and GELU activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        max_dim (int): Dimension to perform max operation over
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = torch.max(x, dim=max_dim, keepdim=True).values
    x = x - x.mean(dim=1, keepdim=True)
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """

    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)

    def forward(self, x, max_dim, fn=module_fn):
        return fn(x, max_dim, self.weight, self.bias)


batch_size = 128
in_features = 512
out_features = 1024
max_dim = 1


def get_inputs():
    return [torch.randn(batch_size, in_features), max_dim]


def get_init_inputs():
    return [in_features, out_features, max_dim]
