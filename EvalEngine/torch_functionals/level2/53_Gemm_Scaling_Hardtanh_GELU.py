import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    scaling_factor: float,
    hardtanh_min: float,
    hardtanh_max: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies GEMM, scaling, hardtanh and GELU activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        scaling_factor (float): Factor to scale the GEMM output
        hardtanh_min (float): Minimum value for hardtanh
        hardtanh_max (float): Maximum value for hardtanh
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor after applying GEMM, scaling, hardtanh and GELU,
            with shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = x * scaling_factor
    x = F.hardtanh(x, min_val=hardtanh_min, max_val=hardtanh_max)
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """

    def __init__(
        self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max
    ):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max,
            self.weight,
            self.bias,
        )


batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
