import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def module_fn(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    block_len: int,
    initial_states: torch.Tensor = None,
) -> torch.Tensor:
    """
    Functional implementation of the SSD operation for Mamba.

    Args:
        X: Input tensor of shape (batch, length, n_heads, d_head)
        A: Parameter tensor of shape (batch, length, n_heads)
        B: Parameter tensor of shape (batch, length, n_heads, d_state)
        C: Parameter tensor of shape (batch, length, n_heads, d_state)
        block_len: Length of each block for chunked computation
        initial_states: Optional initial states

    Returns:
        Final state
    """
    # Rearrange into blocks/chunks
    X_blocks, A_blocks, B_blocks, C_blocks = [
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)
    ]

    A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A_blocks, dim=-1)

    def segsum_fn(x):
        """Naive segment sum calculation."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    # 2. Compute intra-chunk states
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_blocks, decay_states, X_blocks)

    # 3. Compute inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)

    decay_chunk = torch.exp(segsum_fn(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    return new_states[:, -1]


class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.

        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(Model, self).__init__()

        assert (
            seq_length % block_len == 0
        ), "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, fn=module_fn, initial_states=None):
        return fn(X, self.A, self.B, self.C, self.block_len, initial_states)


# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64


def get_inputs():
    return [torch.randn(batch_size, seq_length, n_heads, d_head)]


def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]
