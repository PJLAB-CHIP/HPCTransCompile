#include <torch/extension.h>
#include <omp.h>

// Define the MLP forward function
torch::Tensor mlp_forward_cpu(
    torch::Tensor input,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases) {

    auto device = input.device();
    int num_layers = weights.size();
    auto current = input;

    for (int i = 0; i < num_layers; i++) {
        int batch_size = current.size(0);
        int in_features = current.size(1);
        int out_features = weights[i].size(0);

        auto output = torch::empty({batch_size, out_features}, 
                                     torch::dtype(current.dtype()).device(device));

        // Perform matrix multiplication and bias addition
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < batch_size; row++) {
            for (int col = 0; col < out_features; col++) {
                scalar_t sum = 0;
                for (int k = 0; k < in_features; k++) {
                    sum += current[row][k] * weights[i][col][k];
                }
                sum += biases[i][col];
                output[row][col] = sum;
            }
        }

        // Apply ReLU activation for intermediate layers
        if (i < num_layers - 1) {
            output = torch::clamp(output, 0);
        }

        current = output;
    }

    return current;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mlp_forward_cpu, "MLP forward (CPU)");
}