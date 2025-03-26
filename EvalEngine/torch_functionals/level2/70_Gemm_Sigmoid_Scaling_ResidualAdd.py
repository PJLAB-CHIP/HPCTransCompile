import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """
    Implements Gemm_Sigmoid_Scaling_ResidualAdd pattern using functional operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        weight (torch.Tensor): Weight matrix of shape (hidden_size, input_size)
        bias (torch.Tensor): Bias vector of shape (hidden_size)
        scaling_factor (float): Scaling factor for sigmoid output

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_size)
    """
    x = F.linear(x, weight, bias)
    original_x = x
    x = torch.sigmoid(x)
    x = x * scaling_factor
    x = x + original_x
    return x


class Model(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        gemm = nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias, self.scaling_factor)


batch_size = 128
input_size = 1024
hidden_size = 512
scaling_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
