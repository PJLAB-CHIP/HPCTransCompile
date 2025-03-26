import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> torch.Tensor:
    """
    Functional version of BatchNorm2d

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        weight (torch.Tensor): Weight tensor of shape (num_features).
        bias (torch.Tensor): Bias tensor of shape (num_features).
        running_mean (torch.Tensor): Running mean tensor of shape (num_features).
        running_var (torch.Tensor): Running variance tensor of shape (num_features).
        training (bool): Whether the model is in training mode.
        momentum (float): Momentum parameter for the running mean and variance.
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
    """
    return F.batch_norm(
        x,
        running_mean,
        running_var,
        weight,
        bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


class Model(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """

    def __init__(self, num_features, momentum, eps):
        """
        Initializes the BatchNorm parameters.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, fn=module_fn):
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return fn(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training,
            self.momentum,
            self.eps,
        )


batch_size = 16
features = 64
dim1 = 256
dim2 = 256
momentum = 0.1
eps = 1e-5


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features, momentum, eps]
