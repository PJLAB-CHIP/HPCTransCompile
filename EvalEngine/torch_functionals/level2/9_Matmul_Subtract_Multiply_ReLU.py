import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    subtract_value: float,
    multiply_value: float,
) -> torch.Tensor:
    """
    Applies linear transformation, subtraction, multiplication and ReLU activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        linear_weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        linear_bias (torch.Tensor): Bias vector of shape (out_features)
        subtract_value (float): Value to subtract
        multiply_value (float): Value to multiply

    Returns:
        torch.Tensor: Output tensor after applying linear transformation, subtraction,
            multiplication and ReLU, with shape (batch_size, out_features)
    """
    x = F.linear(x, linear_weight, linear_bias)
    x = x - subtract_value
    x = x * multiply_value
    x = torch.relu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.linear_bias = nn.Parameter(torch.randn(out_features) * 0.02)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.linear_weight,
            self.linear_bias,
            self.subtract_value,
            self.multiply_value,
        )


batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]
