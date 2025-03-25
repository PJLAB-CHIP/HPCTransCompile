#include <torch/extension.h>
#include <vector>
#include <omp.h>

// CPU forward declarations
torch::Tensor fire_forward_cpu(
    torch::Tensor x,
    torch::Tensor squeeze_weight,
    torch::Tensor squeeze_bias,
    torch::Tensor expand1x1_weight,
    torch::Tensor expand1x1_bias,
    torch::Tensor expand3x3_weight,
    torch::Tensor expand3x3_bias
);

// Implement the forward function
torch::Tensor fire_forward_cpu(
    torch::Tensor x,
    torch::Tensor squeeze_weight,
    torch::Tensor squeeze_bias,
    torch::Tensor expand1x1_weight,
    torch::Tensor expand1x1_bias,
    torch::Tensor expand3x3_weight,
    torch::Tensor expand3x3_bias
) {
    // Ensure input tensors are contiguous and on CPU
    x = x.contiguous().cpu();
    squeeze_weight = squeeze_weight.contiguous().cpu();
    squeeze_bias = squeeze_bias.contiguous().cpu();
    expand1x1_weight = expand1x1_weight.contiguous().cpu();
    expand1x1_bias = expand1x1_bias.contiguous().cpu();
    expand3x3_weight = expand3x3_weight.contiguous().cpu();
    expand3x3_bias = expand3x3_bias.contiguous().cpu();

    // Squeeze convolution
    auto x_squeeze = at::conv2d(x, squeeze_weight, squeeze_bias);
    x_squeeze = at::relu(x_squeeze);

    // Expand 1x1 convolution
    auto x1 = at::conv2d(x_squeeze, expand1x1_weight, expand1x1_bias);
    x1 = at::relu(x1);

    // Expand 3x3 convolution with padding
    auto x3 = at::conv2d(x_squeeze, expand3x3_weight, expand3x3_bias, /*stride=*/1, /*padding=*/1);
    x3 = at::relu(x3);

    // Concatenate along channel dimension
    auto output = at::cat({x1, x3}, 1);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fire_forward_cpu, "SqueezeNet Fire Module forward (CPU)");
}