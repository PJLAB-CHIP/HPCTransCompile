#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Host function to perform the linear transformation and accumulation
torch::Tensor my_kernel_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // x: [batch_size, input_size]
    // weight: [hidden_size, input_size]
    // bias: [hidden_size]
    // Output: [batch_size]

    // Get sizes
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size}, options);

    // Configure kernel launch parameters
    int threads = hidden_size;
    int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "my_kernel_forward", ([&] {
        // Parallel loop to compute the linear transformation and accumulation
        #pragma omp parallel for
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            double batch_sum = 0.0;
            for (int hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
                double linear = static_cast<double>(bias[hidden_idx]);
                for (int i = 0; i < input_size; ++i) {
                    linear += static_cast<double>(x[batch_idx * input_size + i]) *
                              static_cast<double>(weight[hidden_idx * input_size + i]);
                }
                // Apply sigmoid function
                double s = 1.0 / (1.0 + exp(-linear));
                batch_sum += s;
            }
            output[batch_idx] = static_cast<scalar_t>(batch_sum);
        }
    }));

    // Reshape output to [batch_size, 1]
    return output.view({batch_size, 1});
}

// PyBind wrapper
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &my_kernel_forward, "Module function forward (CPU)");
}