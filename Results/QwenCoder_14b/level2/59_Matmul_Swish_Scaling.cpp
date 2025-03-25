#include <torch/extension.h>
#include <cmath>
#include <omp.h>

void swish_scaling_cpu_2d(const float* input, float* output, float scaling_factor, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            float x = input[idx];
            // Swish activation: x * sigmoid(x)
            float sigmoid = 1.0f / (1.0f + expf(-x));
            float y = x * sigmoid * scaling_factor;
            output[idx] = y;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double scaling_factor) {

    // Ensure tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Ensure data types are float32
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input tensor 'x' must be of type torch.float32.");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Weight tensor must be of type torch.float32.");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias tensor must be of type torch.float32.");

    // Compute linear transformation: y = x @ weight.T + bias
    auto y = at::addmm(bias, x, weight.t());

    // Get the dimensions
    int rows = y.size(0);
    int cols = y.size(1);

    // Allocate output tensor
    auto output = at::empty_like(y);

    // Apply the CPU version of the kernel
    swish_scaling_cpu_2d(
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(scaling_factor),
        rows,
        cols);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CPU forward function");
}