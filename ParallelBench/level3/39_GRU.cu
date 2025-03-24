#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

__constant__ float ih_consts[2048];
__constant__ float hh_consts[2048];

torch::Tensor forward(
    torch::Tensor x,
    std::vector<torch::Tensor> gru_weights_ih,
    std::vector<torch::Tensor> gru_weights_hh,
    std::vector<torch::Tensor> gru_biases_ih,
    std::vector<torch::Tensor> gru_biases_hh,
    torch::Tensor h0,
    bool is_training) {
    
    h0 = h0.to(x.device());
    
    // Ensure inputs are contiguous for better memory access
    x = x.contiguous();
    h0 = h0.contiguous();
    
    size_t num_layers = gru_weights_ih.size();
    int64_t input_size = x.size(2);
    int64_t hidden_size = gru_weights_hh[0].size(1);
    int64_t seq_length = x.size(0);
    int64_t batch_size = x.size(1);
    
    // Pre-allocate output tensor with optimal memory layout
    auto output = torch::empty({seq_length, batch_size, hidden_size}, 
                             x.options().layout(torch::kStrided)
                             .memory_format(torch::MemoryFormat::Contiguous));
    
    // Create GRU options
    torch::nn::GRUOptions gru_options(input_size, hidden_size);
    gru_options.num_layers(num_layers);
    gru_options.bidirectional(false);
    gru_options.batch_first(false);
    
    auto gru = torch::nn::GRU(gru_options);
    gru->to(x.device());
    gru->train(is_training);
    
    // Pre-process weights and biases for better memory access
    for (size_t l = 0; l < num_layers; ++l) {
        std::string layer_str = std::to_string(l);
        
        // Ensure weights are contiguous and properly aligned
        gru_weights_ih[l] = gru_weights_ih[l].contiguous();
        gru_weights_hh[l] = gru_weights_hh[l].contiguous();
        gru_biases_ih[l] = gru_biases_ih[l].contiguous();
        gru_biases_hh[l] = gru_biases_hh[l].contiguous();
        
        auto params = gru->named_parameters();

        // Copy weights into constant memory if small enough
        if (gru_weights_ih[l].numel() <= 2048 && gru_weights_hh[l].numel() <= 2048) {
            cudaMemcpyToSymbol(ih_consts + l * 2048, gru_weights_ih[l].data_ptr<float>(), gru_weights_ih[l].numel() * sizeof(float));
            cudaMemcpyToSymbol(hh_consts, gru_weights_hh[l].data_ptr<float>(), gru_weights_hh[l].numel() * sizeof(float));
        } else {
            params["weight_ih_l" + layer_str].copy_(gru_weights_ih[l]);
            params["weight_hh_l" + layer_str].copy_(gru_weights_hh[l]);
        }

        params["bias_ih_l" + layer_str].copy_(gru_biases_ih[l]);
        params["bias_hh_l" + layer_str].copy_(gru_biases_hh[l]);
    }
    
    // Reshape h0 with optimal memory layout
    h0 = h0.view({static_cast<int64_t>(num_layers), batch_size, hidden_size});
    
    // Forward pass with optimized memory access
    auto result = gru->forward(x, h0);
    output.copy_(std::get<0>(result));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GRU forward (CUDA)");
}
