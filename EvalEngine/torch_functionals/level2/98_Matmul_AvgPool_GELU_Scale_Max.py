import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    pool_kernel_size: int,
    scale_factor: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Implements Matmul_AvgPool_GELU_Scale_Max pattern using functional operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        pool_kernel_size (int): Kernel size for average pooling
        scale_factor (float): Scale factor to multiply features by
        weight (torch.Tensor): Weight matrix for linear layer
        bias (torch.Tensor): Bias vector for linear layer

    Returns:
        torch.Tensor: Output tensor of shape (batch_size,)
    """
    x = F.linear(x, weight, bias)
    x = F.avg_pool1d(x.unsqueeze(1), kernel_size=pool_kernel_size).squeeze(1)
    x = F.gelu(x)
    x = x * scale_factor
    x = torch.max(x, dim=1).values
    return x


class Model(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """

    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = gemm.weight
        self.bias = gemm.bias

    def forward(self, x, pool_kernel_size, scale_factor, fn=module_fn):
        return fn(x, pool_kernel_size, scale_factor, self.weight, self.bias)


batch_size = 128
in_features = 512
out_features = 256
pool_kernel_size = 4
scale_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_features), pool_kernel_size, scale_factor]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
