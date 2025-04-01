import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

Device = str | torch.device | None

def segsum(x: torch.Tensor, device: Device = None) -> torch.Tensor:
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def module_fn(X, initial_states, A, B, C, chunk_size):
              #batch_size=None, seq_length=None, n_heads=None, d_head=None, d_state=None):
    """Structed State Space Duality (SSD) - the core of Mamba-2"""
    assert X.shape[1] % chunk_size == 0

    # Rearrange into chunks
    X, A, B, C = [rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    
    # Step 1: A 的累积和（归一化与 CUDA 对齐）
    max_a = A.max(dim=-1, keepdim=True)[0]
    #A = A - max_a
    A_cumsum = torch.cumsum(A, dim=-1)
    print(f"Step 1 - A_block[0:5]: {', '.join(f'{x:.6f}' for x in A[0, 0, 0, :5].tolist())}")
    print(f"Step 1 - A_cumsum[0:5]: {', '.join(f'{x:.6f}' for x in A_cumsum[0, 0, 0, :5].tolist())}")
    print(f"Step 1 - prev_cumsum: 0.000000")

    # Step 2: Compute diagonal block outputs
    L = torch.exp(segsum(A, device=X.device))
    print(f"Step 2 - L[0:5]: {', '.join(f'{x:.6f}' for x in L[0, 0, 0, 0, :5].tolist())}")
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    # Debug output
    print("Python A_cumsum[0, 0, 0, :5]:", ', '.join(f'{x:.6f}' for x in A_cumsum[0, 0, 0, :5].tolist()))
    print("Python L[0, 0, 0, 0, :5]:", ', '.join(f'{x:.6f}' for x in L[0, 0, 0, 0, :5].tolist()))
    print("Python Y_diag[0, 0, 0, 0, :5]:", ', '.join(f'{x:.6f}' for x in Y_diag[0, 0, 0, 0, :5].tolist()))

    # Step 3: Compute intra-chunk states
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    print(f"Step 3 - states[0:5]: {', '.join(f'{x:.6f}' for x in states[0, 0, 0, 0, :5].tolist())}")

    # Step 4: Compute inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    print(f"Step 3 - states shape: {states.shape}")
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=X.device))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    
   
    # Step 5: Compute state-to-output conversion
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    # Debugging output
    decay = state_decay_out[0, 0, 0, :5].tolist()
    y_diag = Y_diag[0, 0, 0, 0, :5].tolist()
    y_off = Y_off[0, 0, 0, 0, :5].tolist()
    y_val = Y[0, 0, 0, :5].tolist()
    states_val = states[0, 0, 0, 0, :5].tolist()
    print(f"Step 4 - decay[0:5]: {', '.join(f'{x:.6f}' for x in decay)}")
    print(f"Step 4 - y_diag[0:5]: {', '.join(f'{x:.6f}' for x in y_diag)}")
    print(f"Step 4 - y_off[0:5]: {', '.join(f'{x:.6f}' for x in y_off)}")
    print(f"Step 4 - y_val[0:5]: {', '.join(f'{x:.6f}' for x in y_val)}")
    print(f"Step 4 - states[0:5]: {', '.join(f'{x:.6f}' for x in states_val)}")

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
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def forward(self, X, initial_states=None, fn=module_fn):
        if initial_states is None:
            initial_states = torch.zeros(self.batch_size, 1, self.n_heads, block_len ,self.d_state, device=X.device)
        
        # 统一调用 fn，无论是 Python 还是 CUDA
        return fn(
            X, initial_states, self.A.detach(), self.B.detach(), self.C.detach(), block_len,
            #self.batch_size, self.seq_length, self.n_heads, self.d_head, self.d_state, 
        )

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