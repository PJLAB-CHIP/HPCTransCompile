import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def module_fn(
    x: torch.Tensor,
    params: dict,
    is_training: bool,
) -> torch.Tensor:
    """
    Functional version of the transformer block

    Args:
        x: Input tensor of shape (batch_size, seq_len, n_embd)
        params: Dictionary of parameters
        is_training: Boolean indicating if the model is in training mode

    Returns:
        Output tensor of shape (batch_size, seq_len, n_embd)
    """
    # Layer norm 1
    ln1_out = F.layer_norm(
        x, (params["n_embd"],), params["ln1_weight"], params["ln1_bias"]
    )

    def new_gelu(x):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
        """
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

    def causal_self_attention(
        x,
        c_attn_weight,
        c_attn_bias,
        c_proj_weight,
        c_proj_bias,
        bias,
        n_head,
        attn_dropout_p,
        resid_dropout_p,
        is_training,
    ):
        """
        A vanilla multi-head masked self-attention layer with a projection at the end.
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = F.linear(x, c_attn_weight, c_attn_bias)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if is_training:
            att = F.dropout(att, p=attn_dropout_p, training=True)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = F.linear(y, c_proj_weight, c_proj_bias)
        if is_training:
            y = F.dropout(y, p=resid_dropout_p, training=True)
        return y

    # Self attention
    attn_out = causal_self_attention(
        ln1_out,
        params["c_attn_weight"],
        params["c_attn_bias"],
        params["c_proj_weight"],
        params["c_proj_bias"],
        params["bias"],
        params["n_head"],
        params["attn_pdrop"],
        params["resid_pdrop"],
        is_training,
    )

    x = x + attn_out

    # Layer norm 2
    ln2_out = F.layer_norm(
        x, (params["n_embd"],), params["ln2_weight"], params["ln2_bias"]
    )

    # MLP
    fc_out = F.linear(ln2_out, params["mlp_fc_weight"], params["mlp_fc_bias"])
    act_out = new_gelu(fc_out)
    proj_out = F.linear(act_out, params["mlp_proj_weight"], params["mlp_proj_bias"])
    if is_training:
        proj_out = F.dropout(proj_out, p=params["resid_pdrop"], training=True)

    x = x + proj_out
    return x


class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()

        self.params = nn.ParameterDict()

        # Store config params
        self.params["n_embd"] = n_embd
        self.params["n_head"] = n_head
        self.params["attn_pdrop"] = attn_pdrop
        self.params["resid_pdrop"] = resid_pdrop

        # Layer norms
        ln1 = nn.LayerNorm(n_embd)
        self.params["ln1_weight"] = nn.Parameter(ln1.weight.data.clone())
        self.params["ln1_bias"] = nn.Parameter(ln1.bias.data.clone())

        ln2 = nn.LayerNorm(n_embd)
        self.params["ln2_weight"] = nn.Parameter(ln2.weight.data.clone())
        self.params["ln2_bias"] = nn.Parameter(ln2.bias.data.clone())

        # Attention params
        c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.params["c_attn_weight"] = nn.Parameter(c_attn.weight.data.clone())
        self.params["c_attn_bias"] = nn.Parameter(c_attn.bias.data.clone())

        c_proj = nn.Linear(n_embd, n_embd)
        self.params["c_proj_weight"] = nn.Parameter(c_proj.weight.data.clone())
        self.params["c_proj_bias"] = nn.Parameter(c_proj.bias.data.clone())

        # Causal mask
        self.params["bias"] = nn.Parameter(
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(
                1, 1, max_seqlen, max_seqlen
            ),
            requires_grad=False,
        )

        # MLP params
        c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.params["mlp_fc_weight"] = nn.Parameter(c_fc.weight.data.clone())
        self.params["mlp_fc_bias"] = nn.Parameter(c_fc.bias.data.clone())

        c_proj = nn.Linear(4 * n_embd, n_embd)
        self.params["mlp_proj_weight"] = nn.Parameter(c_proj.weight.data.clone())
        self.params["mlp_proj_bias"] = nn.Parameter(c_proj.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(x, self.params, self.training)


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
