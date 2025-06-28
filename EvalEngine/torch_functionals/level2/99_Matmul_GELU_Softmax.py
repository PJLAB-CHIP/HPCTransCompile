import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies linear transformation, GELU activation, and softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor after applying linear, GELU and softmax,
            with shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = F.gelu(x)
    x = F.softmax(x, dim=1)
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = gemm.weight
        self.bias = gemm.bias

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias)


batch_size = 128
in_features = 100
out_features = 10


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
