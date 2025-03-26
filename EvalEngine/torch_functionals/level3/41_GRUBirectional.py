from typing import List
import torch
import torch.nn as nn
from torch import _VF


def module_fn(
    x: torch.Tensor,
    gru_weights_ih: List[torch.Tensor],
    gru_weights_hh: List[torch.Tensor],
    gru_biases_ih: List[torch.Tensor],
    gru_biases_hh: List[torch.Tensor],
    h0: torch.Tensor,
    is_training: bool,
) -> torch.Tensor:
    """
    Functional implementation of GRU with bidirectional

    Args:
        x: Input tensor of shape (seq_len, batch_size, input_size) if batch_first=False
        gru_weights_ih: List of input-hidden weight tensors for each GRU layer
        gru_weights_hh: List of hidden-hidden weight tensors for each GRU layer
        gru_biases_ih: List of input-hidden bias tensors for each GRU layer
        gru_biases_hh: List of hidden-hidden bias tensors for each GRU layer
        h0: Initial hidden state
        is_training: Whether in training mode

    Returns:
        Output tensor of shape (seq_len, batch_size, num_directions * hidden_size)
    """
    h0 = h0.to(x.device)

    # Collect all parameters for one call
    all_weights = []
    for i in range(len(gru_weights_ih)):
        all_weights.extend(
            [gru_weights_ih[i], gru_weights_hh[i], gru_biases_ih[i], gru_biases_hh[i]]
        )

    # Single call to GRU with all parameters
    output, _ = _VF.gru(
        x,
        h0,
        all_weights,
        True,  # has_biases
        len(gru_weights_ih) // 2,  # num_layers
        0.0,  # dropout
        is_training,  # training
        True,  # bidirectional
        False,
    )  # batch_first

    return output


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

        # Create the original GRU to copy its parameters
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout=0,
            bidirectional=True,
        )

        # Copy h0 exactly as in original code
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))

        # Extract parameters from GRU
        self.gru_weights_ih = nn.ParameterList()
        self.gru_weights_hh = nn.ParameterList()
        self.gru_biases_ih = nn.ParameterList()
        self.gru_biases_hh = nn.ParameterList()

        for i in range(num_layers):
            # Forward direction
            self.gru_weights_ih.append(
                nn.Parameter(getattr(self.gru, f"weight_ih_l{i}").data.clone())
            )
            self.gru_weights_hh.append(
                nn.Parameter(getattr(self.gru, f"weight_hh_l{i}").data.clone())
            )
            self.gru_biases_ih.append(
                nn.Parameter(getattr(self.gru, f"bias_ih_l{i}").data.clone())
            )
            self.gru_biases_hh.append(
                nn.Parameter(getattr(self.gru, f"bias_hh_l{i}").data.clone())
            )

            # Backward direction
            self.gru_weights_ih.append(
                nn.Parameter(getattr(self.gru, f"weight_ih_l{i}_reverse").data.clone())
            )
            self.gru_weights_hh.append(
                nn.Parameter(getattr(self.gru, f"weight_hh_l{i}_reverse").data.clone())
            )
            self.gru_biases_ih.append(
                nn.Parameter(getattr(self.gru, f"bias_ih_l{i}_reverse").data.clone())
            )
            self.gru_biases_hh.append(
                nn.Parameter(getattr(self.gru, f"bias_hh_l{i}_reverse").data.clone())
            )

        # Delete the original GRU to save memory
        del self.gru

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.gru_weights_ih,
            self.gru_weights_hh,
            self.gru_biases_ih,
            self.gru_biases_hh,
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
