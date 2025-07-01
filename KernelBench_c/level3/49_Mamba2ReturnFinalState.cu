#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Define a constant block size for workload distribution
#ifndef FINAL_TILE_SIZE
#define FINAL_TILE_SIZE 256
#endif

// This kernel evenly distributes the workload across threads and blocks.
// Each block is responsible for a tile of the final output for a given (b, h) pair.
// The kernel first cooperatively loads the A_tail_pad data and computes the cumulative sum
// (using a parallel prefix sum) to produce a weight vector. Then, each thread computes one
// element of the final output by iterating over the T dimension.

__global__ void balanced_final_state_kernel(
    const float* __restrict__ A_tail_pad,  // shape: [b, h, T]
    const float* __restrict__ states_cat,    // shape: [b, T, h, p, n]
    float* __restrict__ final_state,         // shape: [b, h, p, n]
    int T,                                   // T = c+1
    int b_size,
    int h_size,
    int p_size,
    int n_size
) {
    // Each block is identified by (b_idx, h_idx, tile_idx)
    int b_idx = blockIdx.x;  // batch index
    int h_idx = blockIdx.y;  // head index
    int tile_idx = blockIdx.z; // tile index for final output elements

    // Use 1D block of threads
    int tid = threadIdx.x;

    // Total number of output elements for final_state for a given (b, h) pair
    int total_outputs = p_size * n_size;
    int base_output = tile_idx * FINAL_TILE_SIZE;
    int global_output = base_output + tid;
    if (global_output >= total_outputs) return;

    // Decode the 2D indices (p, n) from the flattened index
    int p_idx = global_output / n_size;
    int n_idx = global_output % n_size;

    // Allocate shared memory for T floats for cumulative sum and for weights
    // We allocate 2*T floats: first T for s_data, next T for w_data
    extern __shared__ float shared_mem[]; // size = 2 * T * sizeof(float)
    float* s_data = shared_mem;      // cumulative sum storage
    float* w_data = shared_mem + T;  // weight vector storage

    // Load A_tail_pad for the given (b_idx, h_idx) into shared memory.
    // A_tail_pad is of shape [b, h, T] stored in row-major order.
    if (tid < T) {
        int idx = (b_idx * h_size + h_idx) * T + tid;
        s_data[tid] = A_tail_pad[idx];
    }
    __syncthreads();

    // Perform an in-block parallel prefix sum to compute cumulative sum of s_data.
    // Assuming T <= blockDim.x, we use a simple iterative approach.
    for (int offset = 1; offset < T; offset *= 2) {
        float val = 0.0f;
        if (tid >= offset && tid < T) {
            val = s_data[tid - offset];
        }
        __syncthreads();
        if (tid >= offset && tid < T) {
            s_data[tid] += val;
        }
        __syncthreads();
    }

    // The final cumulative sum value is s_data[T-1]
    float s_final = s_data[T - 1];
    // Compute weight for each index: weight = exp(s_final - s_data)
    if (tid < T) {
        w_data[tid] = expf(s_final - s_data[tid]);
    }
    __syncthreads();

    // Each thread computes the dot product for its designated output element
    float sum_val = 0.0f;
    for (int t = 0; t < T; t++) {
        // states_cat shape: [b, T, h, p, n]
        int state_idx = (((b_idx * T + t) * h_size + h_idx) * p_size + p_idx) * n_size + n_idx;
        sum_val += w_data[t] * states_cat[state_idx];
    }
    
    // Write the result to final_state. final_state shape: [b, h, p, n]
    int out_idx = (((b_idx * h_size + h_idx) * p_size + p_idx) * n_size) + n_idx;
    final_state[out_idx] = sum_val;
}

// Forward function that interfaces with PyTorch
// It reshapes and prepares data, then launches the balanced_final_state_kernel

torch::Tensor forward(
    const torch::Tensor& X,       // [b, length, n_heads, d_head]
    const torch::Tensor& A,         // [b, length, n_heads]
    const torch::Tensor& B,         // [b, length, n_heads, d_state]
    const torch::Tensor& C,         // [b, length, n_heads, d_state] (unused)
    int64_t block_len,
    c10::optional<torch::Tensor> initial_states_opt
) {
    // Validate dimensions
    TORCH_CHECK(X.dim() == 4, "X must be [b, length, n_heads, d_head]");
    int b = X.size(0);
    int L = X.size(1);
    int n_heads = X.size(2);
    int dH = X.size(3);  // d_head

    TORCH_CHECK((L % block_len) == 0, "Length must be divisible by block_len");
    int c_chunks = L / block_len;  // number of chunks/blocks

    TORCH_CHECK(B.dim() == 4, "B must be [b, length, n_heads, d_state]");
    int dState = B.size(3);

    // Reshape inputs into blocks
    auto X_blocks = X.reshape({b, c_chunks, block_len, n_heads, dH});       // [b, c_chunks, block_len, n_heads, dH]
    auto A_blocks = A.reshape({b, c_chunks, block_len, n_heads}).permute({0, 3, 1, 2}); // [b, n_heads, c_chunks, block_len]
    auto B_blocks = B.reshape({b, c_chunks, block_len, n_heads, dState});       // [b, c_chunks, block_len, n_heads, dState]
    auto C_blocks = C.reshape({b, c_chunks, block_len, n_heads, dState});       // For consistency

    // Compute cumulative sum and decay states
    auto A_cumsum = A_blocks.cumsum(-1); // [b, n_heads, c_chunks, block_len]
    auto A_last = A_cumsum.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  block_len - 1}).unsqueeze(-1); // [b, n_heads, c_chunks, 1]
    auto decay_states = (A_last - A_cumsum).exp(); // [b, n_heads, c_chunks, block_len]

    // Compute states via einsum: "bclhn,bhcl,bclhp->bchpn"
    auto states = torch::einsum(
        "bclhn,bhcl,bclhp->bchpn",
        {B_blocks, decay_states, X_blocks}
    );

    // Concatenate initial states if provided
    torch::Tensor states_cat;
    if (!initial_states_opt.has_value() || !initial_states_opt.value().defined()) {
        auto init = torch::zeros_like(states.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}));
        states_cat = torch::cat({init, states}, 1); // [b, c_chunks+1, n_heads, dH, dState]
    } else {
        states_cat = torch::cat({initial_states_opt.value(), states}, 1);
    }

    // Prepare A_tail_pad from the last elements of A_cumsum (along block_len) and pad on the left
    auto A_tail = A_cumsum.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  block_len - 1}); // [b, n_heads, c_chunks]
    auto A_tail_pad = torch::constant_pad_nd(A_tail, {1, 0}, 0); // [b, n_heads, c_chunks+1]

    int T = A_tail_pad.size(2);  // T = c_chunks+1
    int b_size = states_cat.size(0);
    int h_size = states_cat.size(2);  // n_heads
    int p_size = states_cat.size(3);  // dH
    int n_size = states_cat.size(4);  // dState

    // Total outputs per (b, h) pair
    int total_outputs = p_size * n_size;
    int grid_z = (total_outputs + FINAL_TILE_SIZE - 1) / FINAL_TILE_SIZE;

    dim3 grid(b_size, h_size, grid_z);
    dim3 block(FINAL_TILE_SIZE);
    size_t shared_mem = 2 * T * sizeof(float);

    auto final_out = torch::empty({b_size, h_size, p_size, n_size}, states_cat.options());

    balanced_final_state_kernel<<<grid, block, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        A_tail_pad.data_ptr<float>(),
        states_cat.data_ptr<float>(),
        final_out.data_ptr<float>(),
        T, b_size, h_size, p_size, n_size
    );
    cudaDeviceSynchronize();

    return final_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SSD-style forward pass with balanced workload distribution (CUDA)");
}
