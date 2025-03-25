#include <torch/extension.h>
#include <vector>
#include <omp.h>

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

    int seq_len = x.size(0);
    int batch_size = x.size(1);
    int hidden_size = h0.size(1);

    // Initialize output tensor
    torch::Tensor output = torch::zeros({seq_len, batch_size, hidden_size}, x.options());

    // Iterate over each sequence step
    for (int t = 0; t < seq_len; ++t) {
        torch::Tensor input_gate = torch::zeros({batch_size, hidden_size}, x.options());
        torch::Tensor reset_gate = torch::zeros({batch_size, hidden_size}, x.options());
        torch::Tensor new_memory = torch::zeros({batch_size, hidden_size}, x.options());

        // Compute gates and new memory for each layer
        for (size_t i = 0; i < weights_ih.size(); ++i) {
            input_gate += torch::mm(x[t], weights_ih[i]) + biases_ih[i];
            reset_gate += torch::mm(x[t], weights_hh[i]) + biases_hh[i];
        }

        // Apply sigmoid and tanh activation functions
        input_gate = torch::sigmoid(input_gate);
        reset_gate = torch::sigmoid(reset_gate);
        new_memory = torch::tanh(new_memory);

        // Update hidden state
        h0 = input_gate * new_memory + (1 - input_gate) * h0;

        // Store the updated hidden state in the output tensor
        output[t] = h0;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gru_forward, "GRU forward (CPU)");
}