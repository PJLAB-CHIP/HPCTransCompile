import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs gemm, swish, divide, clamp, tanh, and clamp operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = x * torch.sigmoid(x)  # Swish activation
    x = x / 2.0
    x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
    x = torch.tanh(x)  # Tanh activation
    x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
    return x


class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        mm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(mm.weight)
        self.bias = nn.Parameter(mm.bias)

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias)


batch_size = 128
in_features = 1024
out_features = 512


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
