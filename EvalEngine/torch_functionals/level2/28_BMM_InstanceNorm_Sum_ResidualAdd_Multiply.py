import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    momentum: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs a linear transform (like batch matrix multiplication), instance normalization,
    summation, residual addition, and final elementwise multiplication, ensuring the behavior
    matches a 2D instance norm usage.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        y (torch.Tensor): Input tensor of shape (batch_size, out_features)
        eps (float): Small constant added to denominator for numerical stability
        momentum (float): Momentum for running stats
        weight (torch.Tensor): Linear layer weights of shape (out_features, in_features)
        bias (torch.Tensor): Linear layer bias of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features).
    """
    # Linear transform (same as nn.Linear but done functionally)
    x = F.linear(x, weight, bias)

    # Reshape to (N, C, H, W) = (batch_size, out_features, 1, 1) to match InstanceNorm2d usage
    x = x.unsqueeze(1).unsqueeze(1)
    # 2D instance normalization
    x = F.instance_norm(
        x,
        None,
        None,
        None,
        None,
        use_input_stats=True,
        momentum=momentum,
        eps=eps,
    )

    # Reshape back to (batch_size, out_features)
    x = x.squeeze(1).squeeze(1)
    # Summation and then elementwise multiplication (residual-like steps)
    x = x + y
    x = x * y

    return x


class Model(nn.Module):
    """
    Model that performs a linear transform, instance normalization, summation, residual addition,
    and multiplication (functionally implemented).
    """

    def __init__(self, in_features, out_features, eps, momentum):
        super(Model, self).__init__()
        # Initialize a linear layer for weights/bias
        bmm = nn.Linear(in_features, out_features)
        # Initialize an InstanceNorm2d layer to borrow weight/bias and track buffers
        instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

        # Expose everything so we can feed them to the functional call
        self.weight = nn.Parameter(bmm.weight)
        self.bias = nn.Parameter(bmm.bias)
        # self.instance_norm_weight = nn.Parameter(instance_norm.weight)
        # self.instance_norm_bias = nn.Parameter(instance_norm.bias)

        # # Buffers to track running statistics
        # self.register_buffer("running_mean", torch.zeros(out_features))
        # self.register_buffer("running_var", torch.ones(out_features))

    def forward(self, x, y, fn=module_fn):
        return fn(
            x,
            y,
            eps,
            momentum,
            self.weight,
            self.bias,
            # self.instance_norm_weight,
            # self.instance_norm_bias,
            # self.running_mean,
            # self.running_var,
        )


batch_size = 128
in_features = 64
out_features = 128
eps = 1e-5
momentum = 0.1


def get_inputs():
    return [
        torch.randn(batch_size, in_features),
        torch.randn(batch_size, out_features),
    ]


def get_init_inputs():
    return [in_features, out_features, eps, momentum]
