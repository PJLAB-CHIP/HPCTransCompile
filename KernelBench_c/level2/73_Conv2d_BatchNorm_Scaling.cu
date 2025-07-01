#include <torch/extension.h>
#include <vector>

torch::Tensor forward(
    torch::Tensor x,
    double scaling_factor,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps,
    double bn_momentum
) {
    // Perform convolution
    x = at::conv2d(x, conv_weight, conv_bias);

    // Perform batch normalization
    x = at::batch_norm(
        x,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_var,
        /*training=*/true,
        /*momentum=*/bn_momentum,
        /*eps=*/bn_eps,
        /*cudnn_enabled=*/at::globalContext().userEnabledCuDNN()
    );

    // Scale the output
    x = x * scaling_factor;

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Performs convolution, batch normalization, and scaling on input tensor");
}