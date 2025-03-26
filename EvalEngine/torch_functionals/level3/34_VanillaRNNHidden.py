import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    i2h_weight: torch.Tensor,
    i2h_bias: torch.Tensor,
    h2o_weight: torch.Tensor,
    h2o_bias: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """
    Vanilla RNN forward pass

    Args:
        x: Input tensor of shape (batch_size, input_size)
        i2h_weight: Weight tensor for input-to-hidden layer
        i2h_bias: Bias tensor for input-to-hidden layer
        h2o_weight: Weight tensor for hidden-to-output layer
        h2o_bias: Bias tensor for hidden-to-output layer
        hidden: Hidden state tensor

    Returns:
        New hidden state tensor
    """
    hidden = hidden.to(x.device)
    combined = torch.cat((x, hidden), dim=1)
    hidden = torch.tanh(F.linear(combined, i2h_weight, i2h_bias))
    output = F.linear(hidden, h2o_weight, h2o_bias)
    return hidden


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.

        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = nn.Parameter(torch.randn((batch_size, hidden_size)))

        # Extract parameters from linear layers
        i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h_weight = nn.Parameter(i2h.weight.data.clone())
        self.i2h_bias = nn.Parameter(i2h.bias.data.clone())

        h2o = nn.Linear(hidden_size, output_size)
        self.h2o_weight = nn.Parameter(h2o.weight.data.clone())
        self.h2o_bias = nn.Parameter(h2o.bias.data.clone())

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(
            x,
            self.i2h_weight,
            self.i2h_bias,
            self.h2o_weight,
            self.h2o_bias,
            self.hidden,
        )


batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size]
