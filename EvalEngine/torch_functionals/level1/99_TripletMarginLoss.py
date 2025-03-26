import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float
) -> torch.Tensor:
    """
    Computes the Triplet Margin Loss for metric learning tasks.

    Args:
        anchor (torch.Tensor): Anchor values.
        positive (torch.Tensor): Positive values.
        negative (torch.Tensor): Negative values.
        margin (float): Margin value.

    Returns:
        torch.Tensor: Triplet Margin Loss.
    """
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin)


class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    """

    def __init__(self, margin):
        super(Model, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, fn=module_fn):
        return fn(anchor, positive, negative, self.margin)


batch_size = 128
input_shape = (4096,)
dim = 1
margin = 1.0


def get_inputs():
    return [
        torch.randn(batch_size, *input_shape),
        torch.randn(batch_size, *input_shape),
        torch.randn(batch_size, *input_shape),
    ]


def get_init_inputs():
    return [margin]
