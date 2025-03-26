import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    eps: float,
    negative_slope: float,
    fc_weight: torch.Tensor,
    fc_bias: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Performs matrix multiplication, group normalization, leaky ReLU and element-wise sum.

    Args:
        x: Input tensor of shape (batch_size, input_size)
        eps: Small constant added to denominator for numerical stability
        negative_slope: Controls negative slope of LeakyReLU
        fc_weight: Weight matrix for linear layer of shape (hidden_size, input_size)
        fc_bias: Bias vector for linear layer of shape (hidden_size)
        gn_weight: Weight parameter for group norm of shape (hidden_size)
        gn_bias: Bias parameter for group norm of shape (hidden_size)
        num_groups: Number of groups for group normalization

    Returns:
        Output tensor of shape (batch_size, hidden_size)
    """
    x = F.linear(x, fc_weight, fc_bias)
    x = F.group_norm(x, num_groups=num_groups, weight=gn_weight, bias=gn_bias, eps=eps)
    x = F.leaky_relu(x, negative_slope=negative_slope)
    x = x + x
    return x


class Model(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """

    def __init__(self, input_size, hidden_size, num_groups, eps, negative_slope):
        super(Model, self).__init__()
        fc = nn.Linear(input_size, hidden_size)
        self.fc_weight = nn.Parameter(fc.weight)
        self.fc_bias = nn.Parameter(fc.bias)
        gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.gn_weight = nn.Parameter(gn.weight + torch.randn(hidden_size) * 0.02)
        self.gn_bias = nn.Parameter(gn.bias + torch.randn(hidden_size) * 0.02)

    def forward(self, x, eps, negative_slope, num_groups, fn=module_fn):
        return fn(
            x,
            eps,
            negative_slope,
            self.fc_weight,
            self.fc_bias,
            self.gn_weight,
            self.gn_bias,
            num_groups,
        )


batch_size = 128
input_size = 512
hidden_size = 256
num_groups = 8
eps = 1e-5
negative_slope = 0.01


def get_inputs():
    return [torch.randn(batch_size, input_size), eps, negative_slope, num_groups]


def get_init_inputs():
    return [input_size, hidden_size, num_groups, eps, negative_slope]
