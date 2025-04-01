import torch
import torch.nn as nn

def module_fn(x, dim):
    exclusive_cumsum = torch.cat((torch.zeros_like(x.select(dim, 0).unsqueeze(dim)), x), dim=dim)[:, :-1]
    return torch.cumsum(exclusive_cumsum, dim=dim)

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=module_fn):
        return fn(x, self.dim)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
