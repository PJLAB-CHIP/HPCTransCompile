import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    scale_factor: float,
    clamp_min: float,
    clamp_max: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies matrix multiplication, scaling, residual connection, clamping, LogSumExp and Mish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        scale_factor (float): Factor to scale the output by
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping
        weight (torch.Tensor): Weight matrix of shape (hidden_size, input_size)
        bias (torch.Tensor): Bias vector of shape (hidden_size)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_size)
    """
    x = F.linear(x, weight, bias)
    x = x * scale_factor
    x = x + x
    x = torch.clamp(x, clamp_min, clamp_max)
    x = torch.logsumexp(x, dim=1, keepdim=True)
    x = x * F.mish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """

    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        matmul = nn.Linear(input_size, hidden_size)
        self.weight = matmul.weight
        self.bias = nn.Parameter(
            matmul.bias + torch.ones_like(matmul.bias) * 0.02
        )  # make sure its nonzero

    def forward(self, x, scale_factor, clamp_min, clamp_max, fn=module_fn):
        return fn(x, scale_factor, clamp_min, clamp_max, self.weight, self.bias)


batch_size = 128
input_size = 512
hidden_size = 1024
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0


def get_inputs():
    return [torch.randn(batch_size, input_size), scale_factor, clamp_min, clamp_max]


def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
