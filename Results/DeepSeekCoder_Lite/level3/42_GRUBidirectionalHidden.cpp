#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <omp.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE_X = 256;
constexpr int BLOCK_SIZE_Y = 4;

__forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __builtin_expect((val + __shfl_down_sync(0xffffffff, val, offset)), 0);
    }
    return val;
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