import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, adds bias, and applies ReLU activation.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix with shape (out_features, in_features)
        bias (torch.Tensor): Bias tensor with shape (out_features,)

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, out_features)
    """
    x = F.linear(x, weight)
    x = x + bias
    x = F.relu(x)
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """

    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias)


batch_size = 128
in_features = 1024
out_features = 512
bias_shape = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]
