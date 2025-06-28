import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    bn_eps: float,
    bn_momentum: float,
    divide_value: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    add_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies matrix multiplication, batch normalization, bias addition, division and Swish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        bn_eps (float): Small constant for numerical stability in batch norm
        bn_momentum (float): Momentum for batch norm running stats
        divide_value (float): Value to divide by
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)
        bn_weight (torch.Tensor): Batch norm weight of shape (out_features)
        bn_bias (torch.Tensor): Batch norm bias of shape (out_features)
        bn_running_mean (torch.Tensor): Batch norm running mean of shape (out_features)
        bn_running_var (torch.Tensor): Batch norm running variance of shape (out_features)
        add_bias (torch.Tensor): Additional bias term of shape (1,)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = F.batch_norm(
        x,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=True,
        momentum=bn_momentum,
        eps=bn_eps,
    )
    x = x + add_bias
    x = x / divide_value
    x = x * torch.sigmoid(x)
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division and Swish activation.
    """

    def __init__(
        self, in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value
    ):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.weight = gemm.weight
        self.bias = gemm.bias
        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.bn_running_mean = nn.Parameter(bn.running_mean, requires_grad=False)
        self.bn_running_var = nn.Parameter(bn.running_var, requires_grad=False)
        self.add_bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, bn_eps, bn_momentum, divide_value, fn=module_fn):
        return fn(
            x,
            bn_eps,
            bn_momentum,
            divide_value,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            self.add_bias,
        )


batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0


def get_inputs():
    return [torch.randn(batch_size, in_features), bn_eps, bn_momentum, divide_value]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
