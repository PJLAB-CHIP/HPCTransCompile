#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Define a constant block size for workload distribution
#ifndef FINAL_TILE_SIZE
#define FINAL_TILE_SIZE 256
#endif

// Function to compute the cumulative sum of an array
void parallel_prefix_sum(float* arr, int size) {
    for (int offset = 1; offset < size; offset *= 2) {
        #pragma omp parallel for
        for (int i = offset; i < size; i++) {
            arr[i] += arr[i - offset];
        }
    }
}

// Function to compute the weight vector
void compute_weights(float* s_data, float* w_data, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        w_data[i] = expf(s_data[size - 1] - s_data[i]);
    }
}

// CPU version of the balanced_final_state_kernel
void balanced_final_state_cpu(
    const float* A_tail_pad,  // shape: [b, h, T]
    const float* states_cat,    // shape: [b, T, h, p, n]
    float* final_state,         // shape: [b, h, p, n]
    int T,                       // T = c+1
    int b_size,
    int h_size,
    int p_size,
    int n_size
) {
    #pragma omp parallel for collapse(3)
    for (int b_idx = 0; b_idx < b_size; b_idx++) {
        for (int h_idx = 0; h_idx < h_size; h_idx++) {
            for (int tile_idx = 0; tile_idx < (p_size * n_size + FINAL_TILE_SIZE - 1) / FINAL_TILE_SIZE; tile_idx++) {
                int base_output = tile_idx * FINAL_TILE_SIZE;
                for (int tid = 0; tid < FINAL_TILE_SIZE; tid++) {
                    int global_output = base_output + tid;
                    if (global_output >= p_size * n_size) continue;

                    int p_idx = global_output / n_size;
                    int n_idx = global_output % n_size;

                    // Allocate shared memory for T floats for cumulative sum and for weights
                    float s_data[T];
                    float w_data[T];

                    // Load A_tail_pad for the given (b_idx, h_idx) into local memory
                    for (int t = 0; t < T; t++) {
                        int idx = (b_idx * h_size + h_idx) * T + t;
                        s_data[t] = A_tail_pad[idx];
                    }

                    // Perform an in-block parallel prefix sum to compute cumulative sum of s_data
                    parallel_prefix_sum(s_data, T);

                    // Compute weight for each index: weight = exp(s_final - s_data)
                    compute_weights(s_data, w_data, T);

                    // Compute the dot product for the designated output element
                    float sum_val = 0.0f;
                    for (int t = 0; t < T; t++) {
                        int state_idx = (((b_idx * T + t) * h_size + h_idx) * p_size + p_idx) * n_size + n_idx;
                        sum_val += w_data[t] * states_cat[state_idx];
                    }

                    // Write the result to final_state
                    int out_idx = (((b_idx * h_size + h_idx) * p_size + p_idx) * n_size) + n_idx;
                    final_state[out_idx] = sum_val;
                }
            }
        }
    }
}

// Forward function that interfaces with PyTorch
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

    auto final_out = torch::empty({b_size, h_size, p_size, n_size}, states_cat.options());

    balanced_final_state_cpu(
        A_tail_pad.data_ptr<float>(),
        states_cat.data_ptr<float>(),
        final_out.data_ptr<float>(),
        T, b_size, h_size, p_size, n_size
    );

    return final_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SSD-style forward pass with balanced workload distribution (CPU)");
}
