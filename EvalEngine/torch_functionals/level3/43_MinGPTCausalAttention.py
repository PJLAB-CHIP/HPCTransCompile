import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def module_fn(
    x: torch.Tensor,
    c_attn_weight: torch.Tensor,
    c_attn_bias: torch.Tensor,
    c_proj_weight: torch.Tensor,
    c_proj_bias: torch.Tensor,
    bias: torch.Tensor,
    n_head: int,
    n_embd: int,
    is_training: bool,
) -> torch.Tensor:
    """
    Functional implementation of MinGPT Causal Attention

    Args:
        x: Input tensor of shape (batch_size, seq_len, n_embd)
        c_attn_weight: Weight tensor for QKV projection
        c_attn_bias: Bias tensor for QKV projection
        c_proj_weight: Weight tensor for output projection
        c_proj_bias: Bias tensor for output projection
        bias: Causal mask tensor
        n_head: Number of attention heads
        n_embd: Embedding dimension
        is_training: Whether in training mode

    Returns:
        Output tensor of shape (batch_size, seq_len, n_embd)
    """
    B, T, C = x.size()

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    qkv = F.linear(x, c_attn_weight, c_attn_bias)
    q, k, v = qkv.split(n_embd, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    if is_training:
        att = F.dropout(att, p=attn_pdrop)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = (
        y.transpose(1, 2).contiguous().view(B, T, C)
    )  # re-assemble all head outputs side by side

    # output projection
    y = F.linear(y, c_proj_weight, c_proj_bias)
    if is_training:
        y = F.dropout(y, p=resid_pdrop)
    return y


class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0

        # Extract parameters from Linear layers
        c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_attn_weight = nn.Parameter(c_attn.weight.data.clone())
        self.c_attn_bias = nn.Parameter(c_attn.bias.data.clone())

        c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj_weight = nn.Parameter(c_proj.weight.data.clone())
        self.c_proj_bias = nn.Parameter(c_proj.bias.data.clone())

        # Register causal mask buffer
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
            self.c_proj_weight,
            self.c_proj_bias,
            self.bias,
            self.n_head,
            self.n_embd,
            self.training,
        )


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0


def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
