#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// CPU function to compute the linear transformation and accumulate the sum
template <typename scalar_t>
void my_cpu_function(
    const scalar_t* x,           // [batch_size, input_size]
    const scalar_t* weight,      // [hidden_size, input_size]
    const scalar_t* bias,        // [hidden_size]
    scalar_t* output,            // [batch_size]
    int input_size,
    int hidden_size) {

    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        double batch_sum = 0.0;
        #pragma omp parallel for reduction(+:batch_sum)
        for (int hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
            // Compute linear transformation
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
}

// Host function to call the CPU function
torch::Tensor my_cpu_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // x: [batch_size, input_size]
    // weight: [hidden_size, input_size]
    // bias: [hidden_size]
    // Output: [batch_size, 1]

    // Get sizes
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size}, options);

    // Call the CPU function
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "my_cpu_forward", ([&] {
        my_cpu_function<scalar_t>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int>(input_size),
            static_cast<int>(hidden_size));
    }));

    // Reshape output to [batch_size, 1]
    return output.view({batch_size, 1});
}

// PyBind wrapper
torch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // Ensure tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Call the CPU function
    return my_cpu_forward(x, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Module function forward (CPU)");
}