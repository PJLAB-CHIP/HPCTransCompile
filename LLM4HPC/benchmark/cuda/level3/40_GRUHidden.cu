#include <torch/extension.h>
#include <vector>

torch::Tensor gru_forward(
    torch::Tensor x,
    std::vector<torch::Tensor> weights_ih,
    std::vector<torch::Tensor> weights_hh,
    std::vector<torch::Tensor> biases_ih,
    std::vector<torch::Tensor> biases_hh,
    torch::Tensor h0,
    bool is_training) {

    // Ensure hidden state is on same device as input
    h0 = h0.to(x.device());

    // Prepare flat weights with contiguous memory
    std::vector<torch::Tensor> flat_weights;
    for (size_t i = 0; i < weights_ih.size(); ++i) {
        flat_weights.push_back(weights_ih[i].contiguous());
        flat_weights.push_back(weights_hh[i].contiguous());
        flat_weights.push_back(biases_ih[i].contiguous());
        flat_weights.push_back(biases_hh[i].contiguous());
    }

    // Call GRU with contiguous weights and proper device alignment
    auto result = torch::gru(
        x,
        h0,
        flat_weights,
        true,             // has_biases
        weights_ih.size(),// num_layers
        0.0,              // dropout
        is_training,      // training
        false,            // bidirectional
        false             // batch_first
    );

    return std::get<1>(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gru_forward, "GRU forward (CUDA)");
}