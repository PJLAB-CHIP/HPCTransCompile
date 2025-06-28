import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    eps: float,
    momentum: float,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    gemm_weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, scaling, and batch normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        eps (float): Small constant for numerical stability in batch norm
        momentum (float): Momentum factor for batch norm running stats
        running_mean (torch.Tensor): Batch norm running mean of shape (out_features,)
        running_var (torch.Tensor): Batch norm running variance of shape (out_features,)
        gemm_weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        gemm_bias (torch.Tensor): Bias vector of shape (out_features,)
        scale (torch.Tensor): Scale factor of shape (out_features,)
        bn_weight (torch.Tensor): Batch norm weight of shape (out_features,)
        bn_bias (torch.Tensor): Batch norm bias of shape (out_features,)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, gemm_weight, gemm_bias)
    x = x * scale
    x = F.batch_norm(
        x,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        training=True,
        momentum=momentum,
        eps=eps,
    )
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """

    def __init__(self, in_features, out_features, scale_shape, eps, momentum):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.gemm_weight = nn.Parameter(gemm.weight)
        self.gemm_bias = nn.Parameter(
            gemm.bias
            + torch.randn(
                gemm.bias.shape, device=gemm.bias.device, dtype=gemm.bias.dtype
            )
            * 0.02
        )
        self.scale = nn.Parameter(torch.randn(scale_shape) * 0.02)
        bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.bn_weight = nn.Parameter(
            bn.weight
            + torch.randn(
                bn.weight.shape, device=bn.weight.device, dtype=bn.weight.dtype
            )
            * 0.02
        )
        self.bn_bias = nn.Parameter(
            bn.bias
            + torch.randn(bn.bias.shape, device=bn.bias.device, dtype=bn.bias.dtype)
            * 0.02
        )
        self.register_buffer("running_mean", torch.randn(out_features))
        self.register_buffer("running_var", torch.abs(torch.randn(out_features)))

    def forward(self, x, eps, momentum, fn=module_fn):
        return fn(
            x,
            eps,
            momentum,
            self.running_mean,
            self.running_var,
            self.gemm_weight,
            self.gemm_bias,
            self.scale,
            self.bn_weight,
            self.bn_bias,
        )


batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)
eps = 1e-5
momentum = 0.1


def get_inputs():
    return [torch.randn(batch_size, in_features), eps, momentum]


def get_init_inputs():
    return [in_features, out_features, scale_shape, eps, momentum]
