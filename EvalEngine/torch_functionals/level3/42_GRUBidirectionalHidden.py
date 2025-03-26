from typing import List
import torch
import torch.nn as nn
from torch import _VF


def module_fn(
    x: torch.Tensor,
    weights_ih_l: List[torch.Tensor],
    weights_hh_l: List[torch.Tensor],
    bias_ih_l: List[torch.Tensor],
    bias_hh_l: List[torch.Tensor],
    h0: torch.Tensor,
    is_training: bool,
) -> torch.Tensor:
    """
    Functional implementation of GRU with bidirectional and hidden state

    Args:
        x: Input tensor of shape (seq_len, batch_size, input_size)
        weights_ih_l: List of input-hidden weights for each layer
        weights_hh_l: List of hidden-hidden weights for each layer
        bias_ih_l: List of input-hidden biases for each layer
        bias_hh_l: List of hidden-hidden biases for each layer
        h0: Initial hidden state
        is_training: Whether in training mode

    Returns:
        h_n: Final hidden state
    """
    h0 = h0.to(x.device)

    # Collect all parameters in the order expected by _VF.gru
    all_weights = []
    num_layers = len(weights_ih_l) // 2
    for layer in range(num_layers):
        # Forward direction
        all_weights.append(weights_ih_l[layer * 2])
        all_weights.append(weights_hh_l[layer * 2])
        all_weights.append(bias_ih_l[layer * 2])
        all_weights.append(bias_hh_l[layer * 2])
        # Backward direction
        all_weights.append(weights_ih_l[layer * 2 + 1])
        all_weights.append(weights_hh_l[layer * 2 + 1])
        all_weights.append(bias_ih_l[layer * 2 + 1])
        all_weights.append(bias_hh_l[layer * 2 + 1])

    # Use the same call signature as nn.GRU
    output, h_n = _VF.gru(
        x,
        h0,
        all_weights,
        True,  # has_biases
        num_layers,  # num_layers
        0.0,  # dropout
        is_training,  # training
        True,  # bidirectional
        False,
    )  # batch_first

    return h_n


class Model(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False
    ):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(Model, self).__init__()

        # Create a GRU instance to get initialized parameters
        gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=0,
            bidirectional=True,
        )

        # Copy the h0 initialization exactly as in original code
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))

        # Extract and store all GRU parameters
        self.weights_ih_l = nn.ParameterList()
        self.weights_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList()
        self.bias_hh_l = nn.ParameterList()

        for i in range(num_layers):
            # Forward direction
            self.weights_ih_l.append(
                nn.Parameter(getattr(gru, f"weight_ih_l{i}").detach())
            )
            self.weights_hh_l.append(
                nn.Parameter(getattr(gru, f"weight_hh_l{i}").detach())
            )
            self.bias_ih_l.append(nn.Parameter(getattr(gru, f"bias_ih_l{i}").detach()))
            self.bias_hh_l.append(nn.Parameter(getattr(gru, f"bias_hh_l{i}").detach()))

            # Backward direction
            self.weights_ih_l.append(
                nn.Parameter(getattr(gru, f"weight_ih_l{i}_reverse").detach())
            )
            self.weights_hh_l.append(
                nn.Parameter(getattr(gru, f"weight_hh_l{i}_reverse").detach())
            )
            self.bias_ih_l.append(
                nn.Parameter(getattr(gru, f"bias_ih_l{i}_reverse").detach())
            )
            self.bias_hh_l.append(
                nn.Parameter(getattr(gru, f"bias_hh_l{i}_reverse").detach())
            )

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.weights_ih_l,
            self.weights_hh_l,
            self.bias_ih_l,
            self.bias_hh_l,
            self.h0,
            self.training,
        )


# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6


def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_layers]
