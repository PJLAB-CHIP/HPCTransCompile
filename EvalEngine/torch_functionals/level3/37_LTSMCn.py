import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF
from typing import List


def module_fn(
    x: torch.Tensor,
    lstm_weights_ih: List[torch.Tensor],
    lstm_weights_hh: List[torch.Tensor],
    lstm_biases_ih: List[torch.Tensor],
    lstm_biases_hh: List[torch.Tensor],
    h0: torch.Tensor,
    c0: torch.Tensor,
    is_training: bool,
) -> torch.Tensor:
    """
    Functional implementation of LSTM with Cn

    Args:
        x: Input tensor of shape (batch_size, sequence_length, input_size)
        lstm_weights_ih: List of input-hidden weight tensors for each LSTM layer
        lstm_weights_hh: List of hidden-hidden weight tensors for each LSTM layer
        lstm_biases_ih: List of input-hidden bias tensors for each LSTM layer
        lstm_biases_hh: List of hidden-hidden bias tensors for each LSTM layer
        h0: Initial hidden state
        c0: Initial cell state
        is_training: Whether in training mode
    Returns:
        Final cell state tensor
    """
    h0 = h0.to(x.device)
    c0 = c0.to(x.device)

    # Run LSTM layers
    out = x
    next_h = h0
    next_c = c0

    for i in range(len(lstm_weights_ih)):
        params = (
            lstm_weights_ih[i],
            lstm_weights_hh[i],
            lstm_biases_ih[i],
            lstm_biases_hh[i],
        )
        result = _VF.lstm(
            out,
            (next_h[i : i + 1], next_c[i : i + 1]),
            params,
            True,  # has_biases
            1,  # num_layers
            0.0 if not is_training else dropout,  # dropout
            is_training,  # training
            False,  # bidirectional
            True,
        )  # batch_first

        out = result[0]
        next_h[i : i + 1] = result[1]
        next_c[i : i + 1] = result[2]

    return next_c


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(Model, self).__init__()

        # Initialize hidden states
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers, batch_size, hidden_size))

        # Extract LSTM parameters
        lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        # Get weights and biases for each layer
        self.lstm_weights_ih = nn.ParameterList()
        self.lstm_weights_hh = nn.ParameterList()
        self.lstm_biases_ih = nn.ParameterList()
        self.lstm_biases_hh = nn.ParameterList()

        for i in range(num_layers):
            self.lstm_weights_ih.append(
                nn.Parameter(getattr(lstm, f"weight_ih_l{i}").data.clone())
            )
            self.lstm_weights_hh.append(
                nn.Parameter(getattr(lstm, f"weight_hh_l{i}").data.clone())
            )
            self.lstm_biases_ih.append(
                nn.Parameter(getattr(lstm, f"bias_ih_l{i}").data.clone())
            )
            self.lstm_biases_hh.append(
                nn.Parameter(getattr(lstm, f"bias_hh_l{i}").data.clone())
            )

        # Extract linear layer parameters
        fc = nn.Linear(hidden_size, output_size)
        self.fc_weight = nn.Parameter(fc.weight.data.clone())
        self.fc_bias = nn.Parameter(fc.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.lstm_weights_ih,
            self.lstm_weights_hh,
            self.lstm_biases_ih,
            self.lstm_biases_hh,
            self.h0,
            self.c0,
            self.training,
        )


# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0


def get_inputs():
    return [torch.randn(batch_size, sequence_length, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
