import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    """
    Performs matrix multiplication, division by scalar, and GELU activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        weight (torch.Tensor): Weight matrix of shape (output_size, input_size)
        bias (torch.Tensor): Bias vector of shape (output_size)
        divisor (float): Scalar divisor

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_size)
    """
    x = F.linear(x, weight, bias)
    x = x / divisor
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """

    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        linear = nn.Linear(input_size, output_size)
        self.weight = linear.weight
        self.bias = linear.bias
        self.divisor = divisor

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias, self.divisor)


batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, output_size, divisor]
