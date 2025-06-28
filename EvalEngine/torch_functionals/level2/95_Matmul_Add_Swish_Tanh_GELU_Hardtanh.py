import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    add_value: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, adds a value, applies Swish, Tanh, GELU and Hardtanh activations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features,)
        add_value (torch.Tensor): Value to add of shape (out_features,)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = x + add_value
    x = torch.sigmoid(x) * x  # Swish
    x = torch.tanh(x)
    x = F.gelu(x)
    x = F.hardtanh(x, min_val=-1, max_val=1)
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """

    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = gemm.weight
        self.bias = gemm.bias
        self.add_value = nn.Parameter(torch.randn(add_value_shape) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias, self.add_value)


batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, add_value_shape]
