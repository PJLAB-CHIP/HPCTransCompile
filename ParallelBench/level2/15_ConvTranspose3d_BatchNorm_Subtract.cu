#include <torch/extension.h>

// Forward declaration of CUDA function
void subtract_mean_cuda(torch::Tensor x);

// The main function equivalent to module_fn in PyTorch
torch::Tensor module_fn(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var) {
    // Transposed convolution
    x = at::conv_transpose3d(
            x, conv_transpose, conv_transpose_bias,
            {stride, stride, stride}, // stride
            {padding, padding, padding} // padding
        );

    // Batch normalization
    bool training = true;
    double momentum = 0.1;
    double eps = 1e-5;
    x = at::batch_norm(
            x,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            training,
            momentum,
            eps,
            /*cudnn_enabled=*/true
        );

    // Mean subtraction over dimensions (2, 3, 4)
    auto mean = x.mean({2, 3, 4}, /*keepdim=*/true);
    x = x - mean;
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Custom module forward function");
}