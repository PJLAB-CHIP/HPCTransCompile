import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies linear transformation followed by two Mish activations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor after linear transformation and two Mish activations,
            with shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = F.mish(x)
    x = F.mish(x)
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight)
        self.bias = nn.Parameter(linear.bias + torch.ones_like(linear.bias) * 0.02)

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias)


batch_size = 128
in_features = 10
out_features = 20


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
