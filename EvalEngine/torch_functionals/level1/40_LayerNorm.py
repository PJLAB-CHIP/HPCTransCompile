import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Functional implementation of LayerNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        weight (torch.Tensor): Weight tensor of shape (normalized_shape).
        bias (torch.Tensor): Bias tensor of shape (normalized_shape).
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    # Get the normalized shape from the weight tensor
    normalized_shape = tuple(x.shape[-len(weight.shape) :])
    return F.layer_norm(
        x, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps
    )


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """

    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer parameters.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).
            fn: Function to apply (defaults to module_fn)

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]
