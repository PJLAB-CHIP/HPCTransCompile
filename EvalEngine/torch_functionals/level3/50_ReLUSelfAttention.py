import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def module_fn(
    x: torch.Tensor,
    c_attn_weight: torch.Tensor,
    c_attn_bias: torch.Tensor,
    bias: torch.Tensor,
    n_head: int,
    n_embd: int,
) -> torch.Tensor:
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.

    Args:
        x: Input tensor of shape (batch_size, seq_len, n_embd)
        c_attn_weight: Weight tensor for QKV projection
        c_attn_bias: Bias tensor for QKV projection
        bias: Causal mask tensor
        n_head: Number of attention heads
        n_embd: Hidden dimension size
        is_training: Whether in training mode

    Returns:
        Output tensor of shape (batch_size, seq_len, n_embd)
    """
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    qkv = F.linear(x, c_attn_weight, c_attn_bias)
    q, k, v = qkv.split(n_embd, dim=2)

    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.relu(att)

    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = (
        y.transpose(1, 2).contiguous().view(B, T, C)
    )  # re-assemble all head outputs side by side
    return y


class Model(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0

        # Keep the original nn.Linear layers
        self.original_c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.original_c_proj = nn.Linear(n_embd, n_embd)

        # Create parameters that share data with the original layers
        self.c_attn_weight = self.original_c_attn.weight
        self.c_attn_bias = self.original_c_attn.bias
        self.c_proj_weight = self.original_c_proj.weight
        self.c_proj_bias = self.original_c_proj.bias

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(
                1, 1, max_seqlen, max_seqlen
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.c_attn_weight,
            self.c_attn_bias,
            self.bias,
            self.n_head,
            self.n_embd,
        )


batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12  # Number of attention heads, typical for BERT-base size


def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
