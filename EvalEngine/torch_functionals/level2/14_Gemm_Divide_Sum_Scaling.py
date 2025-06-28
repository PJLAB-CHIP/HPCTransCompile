import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    scaling_factor: float,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, division, summation and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        scaling_factor (float): Factor to scale the output by
        weight (torch.Tensor): Weight matrix of shape (hidden_size, input_size)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1)
    """
    x = torch.matmul(x, weight.T)  # Gemm
    x = x / 2  # Divide
    x = torch.sum(x, dim=1, keepdim=True)  # Sum
    x = x * scaling_factor  # Scaling
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size) * 0.02)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.scaling_factor, self.weight)


batch_size = 128
input_size = 10
hidden_size = 20
scaling_factor = 1.5


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
