import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, applies sigmoid, and sums the result.

    Args:
        x: Input tensor of shape (batch_size, input_size)
        weight: Weight tensor of shape (hidden_size, input_size)
        bias: Bias tensor of shape (hidden_size)

    Returns:
        Output tensor of shape (batch_size, 1)
    """
    x = F.linear(x, weight, bias)
    x = torch.sigmoid(x)
    x = torch.sum(x, dim=1, keepdim=True)
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        gemm = nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)

    def forward(self, x, fn=module_fn):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        return fn(x, self.weight, self.bias)


batch_size = 128
input_size = 10
hidden_size = 20


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]
