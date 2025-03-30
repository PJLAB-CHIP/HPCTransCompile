#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double scaling_factor) {

    // Ensure tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Ensure tensors are on CUDA
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor.");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor.");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor.");

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

    // Parallelize the swish scaling operation
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            float x_val = y[row * cols + col].item<float>();
            // Swish activation: x * sigmoid(x)
            float sigmoid = 1.0f / (1.0f + expf(-x_val));
            float y_val = x_val * sigmoid * static_cast<float>(scaling_factor);
            output[idx] = y_val;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CPU forward function");
}