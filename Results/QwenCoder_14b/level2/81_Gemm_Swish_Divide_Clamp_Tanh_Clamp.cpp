#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <omp.h>

// Function to perform the operations in the CUDA kernel
void module_kernel_cpu(
    const float* x_in,
    float* x_out,
    int height,
    int width) {

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int index = row * width + col;
            float x = x_in[index];

            // Swish activation: x = x * sigmoid(x)
            float sigmoid_x = 1.0f / (1.0f + exp(-x));
            x = x * sigmoid_x;

            // Divide by 2
            x = x / 2.0f;

            // Clamp between -1 and 1
            x = std::max(std::min(x, 1.0f), -1.0f);

            // Tanh activation
            x = tanhf(x);

            // Clamp again between -1 and 1
            x = std::max(std::min(x, 1.0f), -1.0f);

            x_out[index] = x;
        }
    }
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

    module_kernel_cpu(
        x_linear.data_ptr<float>(),
        x_out.data_ptr<float>(),
        height,
        width);

    return x_out;
}

// C++ interface
torch::Tensor module_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
    return module_forward_cpu(x, weight, bias);
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_forward, "Custom module forward function (CPU)");
}