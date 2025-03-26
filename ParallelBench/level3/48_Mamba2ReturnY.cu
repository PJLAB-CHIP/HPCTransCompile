```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper function for cumulative sum
template<typename scalar_t>
__global__ void cumsum_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size,
    int seq_len,
    int n_heads) {
    
    int b = blockIdx.y;
    int h = blockIdx.z;
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (b >= batch_size || h >= n_heads || t >= seq_len) return;
    
    int idx = b * seq_len * n_heads + t * n_heads + h;
    
    scalar_t sum = 0;
    for (int i = 0; i <= t; i++) {
        int curr_idx = b * seq_len * n_heads + i * n_heads + h;
        sum += input[curr_idx];
    }
    output[idx] = sum;
}

// Segment sum calculation
template<typename scalar_t>
__global__ void segsum_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size,
    int seq_len,
    int n_heads) {
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = threadIdx.y + blockIdx.x * blockDim.y;
    int j = threadIdx.x;
    
    if (b >= batch_size || h >= n_heads || i >= seq_len || j >= seq_len) return;
    
    int input_idx = b * seq_len * n_heads + i * n_heads + h;
    int output_idx = b * seq_len * seq_len * n_heads + i * seq_len * n_heads + j * n_heads + h;
    
    if (j <= i) {
        scalar_t val_i = input[input_idx];
        scalar_t val_j = (j > 0) ? input[input_idx - (j * n_heads)] : 0;
        output[output_idx] = val_i - val_j;
    } else {
        output[output_idx] = -INFINITY;
    }
}

// Main module function
torch::Tensor module_forward(
    torch::Tensor X,
    torch::Tensor initial_states,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int batch_size,
    int seq_length,
    int n_heads,
    int d_head,
    int d_state,
    int block_len) {
    
    // Input checks
    CHECK_INPUT(X);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);
    if (initial_states.defined()) CHECK_INPUT(initial_states);
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    
    // Rearrange tensors into blocks
    int n_chunks = seq_length / block_len;
    auto X_blocks = X.view({batch_size, n_chunks, block_len, n_heads, d_head});
    auto A_blocks = A.view({batch_size, n_chunks, block_len, n_heads}).permute({0, 3, 1, 2});
    auto B_blocks = B.view({batch_size, n_chunks, block_len, n_heads, d_state});
    auto C_blocks = C.view({batch_size, n_chunks, block_len, n_heads, d_state});
    
    // Compute A_cumsum
    auto A_cumsum = torch::cumsum(A_blocks, -1);
    
    // 1. Compute diagonal block outputs (L and Y_diag)
    auto options = torch::TensorOptions().dtype(X.dtype()).device(X.device());
    auto L = torch::empty({batch_size, n_heads, n_chunks, block_len, block_len}, options);
    
    dim3 segsum_grid((block_len + 15) / 16, n_heads, batch_size);
    dim3 segsum_block(16, 16);
    
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "segsum_kernel", ([&] {
        segsum_kernel<scalar_t><<<segsum_grid, segsum_block>>>(
            A_blocks.data_ptr<scalar_t>(),
            L.data_ptr<scalar_t>(),
            batch_size,
            block_len,
            n_heads);
    }));
    
    L = torch::exp(L);
    auto Y_diag = torch::einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                               {C_blocks, B_blocks, L, X_blocks});
    
    // 2. Compute intra-chunk states
    auto decay_states = torch::exp((A_cumsum.index({"...", -1}).unsqueeze(-1) - A_cumsum));
    auto states = torch::einsum("bclhn,bhcl,bclhp->bchpn", 
                              {B_blocks, decay_states, X_blocks});
    
    // 3. Compute inter-chunk recurrence
    if (!initial_states.defined()) {
        initial_states = torch::zeros({batch_size, 1, n_heads, d_state, d_head}, options);
    }
    states = torch::cat({initial_states, states}, 1);
    
    auto A_last = A_cumsum.index({"...", -1});
    auto A_padded = F::pad(A_last, {1, 0});
    auto decay_chunk = torch::exp(segsum(A_padded));
    
    auto new_states = torch::einsum("bhzc,bchpn->bzhpn", {decay_chunk, states});
    states = new_states.index({"...", Slice(None, -1)});
    
    // 4. Compute state-to-output conversion
    auto state_decay_out = torch::exp(A_cumsum);
    auto Y_off = torch::einsum("bclhn,bchpn,bhcl->bclhp", 
                             {C_blocks, states, state_decay_out});
    
    // Combine results
    auto Y = Y_diag + Y_off;
    Y = Y.view({batch_size, seq_length, n_heads, d_head});
    
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_forward, "Module forward pass");
}