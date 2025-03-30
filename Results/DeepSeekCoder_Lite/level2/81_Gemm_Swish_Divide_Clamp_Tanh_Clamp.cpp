#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

// Macros for input checking
#define CHECK_INPUT(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")

// CPU implementation of the Swish, Sigmoid, Tanh activation functions
template <typename scalar_t>
scalar_t sigmoid(scalar_t x) {
    return static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + exp(-x));
}

template <typename scalar_t>
scalar_t tanh_activation(scalar_t x) {
    return tanh(x);
}

// CPU forward function
torch::Tensor module_forward_cpu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Execute linear operation: x_linear = F.linear(x, weight, bias)
    auto x_linear = torch::addmm(bias, x, weight.t());
    auto x_out = torch::empty_like(x_linear);

    // Assuming x_linear is a 2D matrix
    int height = x_linear.size(0);
    int width = x_linear.size(1);

    // Parallelize over the height and width of the matrix
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int index = row * width + col;
            scalar_t x_val = x_linear.data_ptr<scalar_t>()[index];

            // Swish activation: x = x * sigmoid(x)
            scalar_t sigmoid_x = sigmoid(x_val);
            x_val = x_val * sigmoid_x;

            // Divide by 2
            x_val = x_val / static_cast<scalar_t>(2);

            // Clamp between -1 and 1
            x_val = std::max(std::min(x_val, static_cast<scalar_t>(1)), static_cast<scalar_t>(-1));

            // Tanh activation
            x_val = tanh_activation(x_val);

            // Clamp again between -1 and 1
            x_val = std::max(std::min(x_val, static_cast<scalar_t>(1)), static_cast<scalar_t>(-1));

            x_out.data_ptr<scalar_t>()[index] = x_val;
        }
    }

    return x_out;
}

// C++ interface
torch::Tensor module_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    return module_forward_cpu(x, weight, bias);
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_forward, "Custom module forward function (CPU)");
}