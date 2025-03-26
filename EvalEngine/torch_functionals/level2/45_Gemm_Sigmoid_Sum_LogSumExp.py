import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    linear1_weight: torch.Tensor,
    linear1_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, applies Sigmoid, sums result, and calculates LogSumExp.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        linear1_weight (torch.Tensor): Weight matrix for first linear layer of shape (hidden_size, input_size)
        linear1_bias (torch.Tensor): Bias vector for first linear layer of shape (hidden_size)

    Returns:
        torch.Tensor: Scalar output after applying linear layers, sigmoid, sum and logsumexp
    """
    x = F.linear(x, linear1_weight, linear1_bias)
    x = torch.sigmoid(x)
    x = torch.sum(x, dim=1)
    x = torch.logsumexp(x, dim=0)
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid, sums the result, and calculates the LogSumExp.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        lin1 = nn.Linear(input_size, hidden_size)
        self.linear1_weight = nn.Parameter(lin1.weight)
        self.linear1_bias = nn.Parameter(
            lin1.bias
            + torch.randn(
                lin1.bias.shape, device=lin1.bias.device, dtype=lin1.bias.dtype
            )
            * 0.02
        )

    def forward(self, x, fn=module_fn):
        return fn(x, self.linear1_weight, self.linear1_bias)


batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size]
