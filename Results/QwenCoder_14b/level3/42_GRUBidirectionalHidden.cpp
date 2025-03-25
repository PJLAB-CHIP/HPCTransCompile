#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <omp.h>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE_X = 256;
constexpr int BLOCK_SIZE_Y = 4;

float warp_reduce_sum(float val) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += val;
    }
    return val;
}

void gru_bidirectional_cpu(
    const float* input,
    const float* weights,
    const float* hidden,
    float* output,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int direction) {
    
    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
            const int effective_seq = direction == 0 ? seq_idx : (seq_length - 1 - seq_idx);
            
            if (batch_idx >= batch_size || seq_idx >= seq_length) continue;
            
            float local_sum = 0.0f;
            
            #pragma omp parallel for reduction(+:local_sum)
            for (int h = 0; h < hidden_size; ++h) {
                const int input_idx = batch_idx * seq_length * hidden_size + 
                                    effective_seq * hidden_size + h;
                const int weight_idx = h * hidden_size;
                
                float inp = input[input_idx];
                float w = weights[weight_idx];
                local_sum += inp * w;
            }
            
            local_sum = warp_reduce_sum(local_sum);
            
            const int output_idx = batch_idx * seq_length * hidden_size + 
                                 effective_seq * hidden_size;
            output[output_idx] = local_sum;
        }
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