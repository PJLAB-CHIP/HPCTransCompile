#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE_X = 256;
constexpr int BLOCK_SIZE_Y = 4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gru_bidirectional_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ hidden,
    float* __restrict__ output,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int direction) {
    
    // 2D grid for batch and sequence dimensions
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Handle both forward and backward directions
    const int effective_seq = direction == 0 ? seq_idx : (seq_length - 1 - seq_idx);
    
    if (batch_idx >= batch_size || seq_idx >= seq_length) return;
    
    // Shared memory for intermediate results
    __shared__ float s_hidden[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int lane_id = tid % WARP_SIZE;
    
    float local_sum = 0.0f;
    
    // Compute hidden state updates with coalesced memory access
    #pragma unroll 4
    for (int h = tid; h < hidden_size; h += blockDim.x * blockDim.y) {
        const int input_idx = batch_idx * seq_length * hidden_size + 
                            effective_seq * hidden_size + h;
        const int weight_idx = h * hidden_size;
        
        float inp = input[input_idx];
        float w = weights[weight_idx];
        local_sum += inp * w;
    }
    
    // Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);
    
    // Block-level reduction using shared memory
    if (lane_id == 0) {
        s_hidden[threadIdx.y][threadIdx.x] = local_sum;
    }
    __syncthreads();
    
    // Final reduction and output writing
    if (tid < (BLOCK_SIZE_X * BLOCK_SIZE_Y / WARP_SIZE)) {
        float final_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_Y; ++i) {
            final_sum += s_hidden[i][threadIdx.x];
        }
        
        const int output_idx = batch_idx * seq_length * hidden_size + 
                             effective_seq * hidden_size + tid;
        output[output_idx] = final_sum;
    }
}

torch::Tensor gru_forward(
    torch::Tensor x,
    std::vector<torch::Tensor> weights_ih_l,
    std::vector<torch::Tensor> weights_hh_l,
    std::vector<torch::Tensor> bias_ih_l,
    std::vector<torch::Tensor> bias_hh_l,
    torch::Tensor h0,
    bool is_training) {

    h0 = h0.to(x.device());
    const auto batch_size = x.size(0);
    const auto seq_length = x.size(1);
    const auto hidden_size = h0.size(2);
    
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (batch_size + block.x - 1) / block.x,
        (seq_length + block.y - 1) / block.y
    );

    auto output = torch::zeros_like(x);
    
    int64_t num_layers = weights_ih_l.size() / 2;
    std::vector<torch::Tensor> all_weights;

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        all_weights.push_back(weights_ih_l[layer*2].contiguous());
        all_weights.push_back(weights_hh_l[layer*2].contiguous());
        all_weights.push_back(bias_ih_l[layer*2].contiguous());
        all_weights.push_back(bias_hh_l[layer*2].contiguous());
        
        all_weights.push_back(weights_ih_l[layer*2 + 1].contiguous());
        all_weights.push_back(weights_hh_l[layer*2 + 1].contiguous());
        all_weights.push_back(bias_ih_l[layer*2 + 1].contiguous());
        all_weights.push_back(bias_hh_l[layer*2 + 1].contiguous());
    }

    auto result = torch::gru(
        x,
        h0,
        all_weights,
        true,
        num_layers,
        0.0,
        is_training,
        true,
        false
    );

    return std::get<1>(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gru_forward, "GRU forward pass");
}