import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF

def lstm_module_fn(input, hx, params, num_layers, dropout, bidirectional):
    weight_ih_l0 = params['weight_ih_l0']
    weight_hh_l0 = params['weight_hh_l0']
    bias_ih_l0 = params['bias_ih_l0']
    bias_hh_l0 = params['bias_hh_l0']
    
    all_weights = []
    for layer in range(num_layers):
        for direction in range(2 if bidirectional else 1):
            suffix = '_reverse' if direction == 1 else ''
            layer_input_size = input.size(-1) if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            w_ih = params[f'weight_ih_l{layer}{suffix}']
            w_hh = params[f'weight_hh_l{layer}{suffix}']
            b_ih = params[f'bias_ih_l{layer}{suffix}']
            b_hh = params[f'bias_hh_l{layer}{suffix}']
            
            all_weights.append([w_ih, w_hh, b_ih, b_hh])
    
    output, (hn, cn) = _VF.lstm(input, hx, all_weights, True, num_layers, dropout, False, bidirectional)
    return output, (hn, cn)

def linear_module_fn(input, weight, bias):
    return F.linear(input, weight, bias)

def module_fn(x, h0, c0, params, num_layers, dropout, output_size, hidden_size, bidirectional=True):
    # Forward propagate LSTM
    out, (hn, cn) = lstm_module_fn(x, (h0, c0), params['lstm'], num_layers, dropout, bidirectional)
    
    # Decode the hidden state of the last time step
    out = linear_module_fn(out[:, -1, :], params['fc']['weight'], params['fc']['bias'])
    
    return out

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.output_size = output_size
        
        # Initialize hidden state with random values
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        
        # Extract parameters from LSTM and Linear modules
        lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        fc = nn.Linear(hidden_size * 2, output_size)
        
        # Store parameters in a dictionary
        self.params = nn.ParameterDict()
        
        # LSTM parameters
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name or 'bias_ih' in name or 'bias_hh' in name:
                self.params['lstm.' + name.replace('.', '_')] = nn.Parameter(param.data.clone())
        
        # FC parameters
        self.params['fc.weight'] = nn.Parameter(fc.weight.data.clone())
        self.params['fc.bias'] = nn.Parameter(fc.bias.data.clone())
    
    def forward(self, x, fn=module_fn):
        self.h0 = self.h0.to(x.device)
        self.c0 = self.c0.to(x.device)
        
        # Prepare parameters for functional call
        params = {
            'lstm': {k.replace('lstm_', '').replace('.', '_'): v for k, v in self.params.items() if 'lstm' in k},
            'fc': {'weight': self.params['fc.weight'], 'bias': self.params['fc.bias']}
        }
        
        return fn(x, self.h0, self.c0, params, self.num_layers, self.dropout, self.output_size, self.hidden_size)

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