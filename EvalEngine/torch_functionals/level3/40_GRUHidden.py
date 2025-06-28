from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF


def module_fn(
    x: torch.Tensor,
    weights_ih: List[torch.Tensor],
    weights_hh: List[torch.Tensor],
    biases_ih: List[torch.Tensor],
    biases_hh: List[torch.Tensor],
    h0: torch.Tensor,
    is_training: bool,
) -> torch.Tensor:
    """
    Functional implementation of GRU with hidden state

    Args:
        x: Input tensor of shape (seq_len, batch_size, input_size) if batch_first=False
        weights_ih: List of input-hidden weight tensors for each GRU layer
        weights_hh: List of hidden-hidden weight tensors for each GRU layer
        biases_ih: List of input-hidden bias tensors for each GRU layer
        biases_hh: List of hidden-hidden bias tensors for each GRU layer
        h0: Initial hidden state
        is_training: Whether in training mode

    Returns:
        h_n: Final hidden state
    """
    h0 = h0.to(x.device)

    # Run GRU layer
    flat_weights = []
    for i in range(len(weights_ih)):
        flat_weights.append(weights_ih[i])
        flat_weights.append(weights_hh[i])
        flat_weights.append(biases_ih[i])
        flat_weights.append(biases_hh[i])

    output, h_n = _VF.gru(
        x,
        h0,
        flat_weights,
        True,  # has_biases
        len(weights_ih),  # num_layers
        0.0,  # dropout
        is_training,  # training
        False,  # bidirectional
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

        # Create a GRU to get its parameters
        gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout=0,
            bidirectional=False,
        )

        # Initialize hidden state exactly as in original
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))

        # Extract and register GRU parameters
        self.weights_ih = nn.ParameterList()
        self.weights_hh = nn.ParameterList()
        self.biases_ih = nn.ParameterList()
        self.biases_hh = nn.ParameterList()

        # Use the same parameter initialization as nn.GRU
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size

            # Get the parameters from the GRU module
            w_ih = getattr(gru, f"weight_ih_l{i}")
            w_hh = getattr(gru, f"weight_hh_l{i}")
            b_ih = getattr(gru, f"bias_ih_l{i}")
            b_hh = getattr(gru, f"bias_hh_l{i}")

            # Register them as parameters
            self.weights_ih.append(nn.Parameter(w_ih.data))
            self.weights_hh.append(nn.Parameter(w_hh.data))
            self.biases_ih.append(nn.Parameter(b_ih.data))
            self.biases_hh.append(nn.Parameter(b_hh.data))

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.weights_ih,
            self.weights_hh,
            self.biases_ih,
            self.biases_hh,
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
