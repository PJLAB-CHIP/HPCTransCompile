import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def segsum(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def module_fn(X, initial_states, A, B, C, batch_size, seq_length, n_heads, d_head, d_state, block_len):
    # Rearrange into blocks/chunks
    X_blocks = rearrange(X, "b (c l) h p -> b c l h p", l=block_len)
    A_blocks = rearrange(A, "b (c l) h -> b h c l", l=block_len)
    B_blocks = rearrange(B, "b (c l) h s -> b c l h s", l=block_len)
    C_blocks = rearrange(C, "b (c l) h s -> b c l h s", l=block_len)
    
    A_cumsum = torch.cumsum(A_blocks, dim=-1)
    
    # 1. Compute diagonal block outputs
    L = torch.exp(segsum(A_blocks))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                         C_blocks, B_blocks, L, X_blocks)
    
    # 2. Compute intra-chunk states
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                        B_blocks, decay_states, X_blocks)
    
    # 3. Compute inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]
    
    # 4. Compute state-to-output conversion
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                       C_blocks, states, state_decay_out)
    
    # Combine diagonal and off-diagonal terms
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    
    return Y

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
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
        
    def forward(self, X, initial_states=None, fn=module_fn):
        return fn(X, initial_states, self.A, self.B, self.C, self.batch_size, self.seq_length, self.n_heads, self.d_head, self.d_state, self.block_len)

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