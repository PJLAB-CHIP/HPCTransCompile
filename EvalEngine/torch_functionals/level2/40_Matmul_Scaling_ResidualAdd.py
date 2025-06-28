import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    scaling_factor: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, scaling, and residual addition.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        scaling_factor (float): Scaling factor to apply after matrix multiplication
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    original_x = x.clone().detach()
    x = x * scaling_factor
    x = x + original_x
    return x


class Model(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        scaling_factor (float): Scaling factor to apply after matrix multiplication
    """

    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        mm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(mm.weight)
        self.bias = nn.Parameter(
            mm.bias
            + torch.randn(mm.bias.shape, device=mm.bias.device, dtype=mm.bias.dtype)
            * 0.02
        )
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.scaling_factor, self.weight, self.bias)


batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
