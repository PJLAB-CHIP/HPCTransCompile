import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract: torch.Tensor,
) -> torch.Tensor:
    """
    Performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix for linear layer of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector for linear layer of shape (out_features)
        subtract (torch.Tensor): Vector to subtract of shape (out_features)

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    original_x = x.clone().detach()

    # Gemm
    x = F.linear(x, weight, bias)

    # Subtract
    x = x - subtract

    # GlobalAvgPool
    x = torch.mean(x, dim=1, keepdim=True)

    # LogSumExp
    x = torch.logsumexp(x, dim=1, keepdim=True)

    # GELU
    x = F.gelu(x)

    # ResidualAdd
    x = x + original_x

    return x


class Model(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)
        self.subtract = nn.Parameter(torch.randn(out_features) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias, self.subtract)


batch_size = 128
in_features = 1024
out_features = 512


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
