#include <torch/extension.h>
#include <vector>
#include <algorithm>

// Function to perform matrix multiplication and addition with bias
template <typename scalar_t>
void mlp_forward_cpu(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const scalar_t* bias) {

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            scalar_t sum = 0;
            const int input_offset = row * in_features;
            const int weight_offset = col * in_features;

            // Loop over in_features with 4x unrolling
            const int total = in_features;
            const int stride = 4;
            for (int i = 0; i < total; i += stride) {
                scalar_t temp = 0;
                if (i + 3 < total) {
                    temp = input[input_offset + i]     * weight[weight_offset + i] +
                           input[input_offset + i + 1] * weight[weight_offset + i + 1] +
                           input[input_offset + i + 2] * weight[weight_offset + i + 2] +
                           input[input_offset + i + 3] * weight[weight_offset + i + 3];
                } else {
                    for (int j = 0; j < 4 && (i + j) < total; j++) {
                        temp += input[input_offset + i + j] * weight[weight_offset + i + j];
                    }
                }
                sum += temp;
            }

            // Add bias
            sum += bias[col];
            output[row * out_features + col] = sum;
        }
    }
}

// Function to apply ReLU activation
template <typename scalar_t>
void relu_cpu(
    scalar_t* data,
    const int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

// Host function chaining layers: performing matrix multiplication and ReLU between layers
torch::Tensor mlp_cpu_forward(
    torch::Tensor input,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> biases) {

    auto device = input.device();
    int num_layers = weights.size();
    auto current = input;

    for (int i = 0; i < num_layers; ++i) {
        int batch_size = current.size(0);
        int in_features = current.size(1);
        int out_features = weights[i].size(0);

        auto output = torch::empty({batch_size, out_features}, 
                                     torch::dtype(current.dtype()).device(device));

        AT_DISPATCH_FLOATING_TYPES(current.scalar_type(), "mlp_forward_cpu", ([&] {
            mlp_forward_cpu<scalar_t>(
                current.data_ptr<scalar_t>(),
                weights[i].data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_features,
                out_features,
                biases[i].data_ptr<scalar_t>()
            );
        }));

        // Apply ReLU activation for intermediate layers
        if (i < num_layers - 1) {
            int size = batch_size * out_features;
            AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "relu_cpu", ([&] {
                relu_cpu<scalar_t>(
                    output.data_ptr<scalar_t>(),
                    size
                );
            }));
        }

        current = output;
    }

    return current;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mlp_cpu_forward, "MLP forward (CPU)");
}
