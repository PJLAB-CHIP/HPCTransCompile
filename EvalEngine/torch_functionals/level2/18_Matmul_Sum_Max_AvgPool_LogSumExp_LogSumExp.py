import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1)
    """
    x = F.linear(x, weight, bias)  # (batch_size, out_features)
    x = torch.sum(x, dim=1, keepdim=True)  # (batch_size, 1)
    x = torch.max(x, dim=1, keepdim=True)[0]  # (batch_size, 1)
    x = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1)
    x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
    x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
    return x


class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        linear = nn.Linear(in_features, out_features)
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias)


batch_size = 128
in_features = 10
out_features = 5


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
