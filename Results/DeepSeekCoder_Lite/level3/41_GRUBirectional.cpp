#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor gru_forward(
    torch::Tensor x,
    std::vector<torch::Tensor> gru_weights_ih,
    std::vector<torch::Tensor> gru_weights_hh,
    std::vector<torch::Tensor> gru_biases_ih,
    std::vector<torch::Tensor> gru_biases_hh,
    torch::Tensor h0,
    bool is_training) {

    // Ensure h0 is on the same device as input tensor x
    h0 = h0.to(x.device());

    // Prepare all_weights list matching PyTorch's expected format
    std::vector<torch::Tensor> all_weights;
    for (size_t i = 0; i < gru_weights_ih.size(); ++i) {
        // Ensure weights are on the same device as input
        gru_weights_ih[i] = gru_weights_ih[i].to(x.device());
        gru_weights_hh[i] = gru_weights_hh[i].to(x.device());
        gru_biases_ih[i] = gru_biases_ih[i].to(x.device());
        gru_biases_hh[i] = gru_biases_hh[i].to(x.device());
        
        all_weights.push_back(gru_weights_ih[i]);
        all_weights.push_back(gru_weights_hh[i]);
        all_weights.push_back(gru_biases_ih[i]);
        all_weights.push_back(gru_biases_hh[i]);
    }

    // Calculate num_layers from bidirectional setup
    int num_layers = gru_weights_ih.size() / 2;

    // Call optimized GRU implementation
    auto result = torch::gru(
        x,
        h0,
        all_weights,
        true,        // has_biases
        num_layers,  // num_layers
        0.0,         // dropout
        is_training, // training
        true,        // bidirectional
        false        // batch_first
    );

    return std::get<0>(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gru_forward, "GRU forward (CPU)");
}