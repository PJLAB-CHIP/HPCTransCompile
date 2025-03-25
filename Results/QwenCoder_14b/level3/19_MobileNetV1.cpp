#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp.h>

using namespace at;
namespace py = pybind11;

// Function to perform convolution on CPU
torch::Tensor conv2d_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    // Implement CPU version of conv2d
    // This is a placeholder for the actual implementation
    return at::conv2d(input, weight, c10::nullopt, stride, padding, dilation, groups);
}

// Function to perform batch normalization on CPU
torch::Tensor batch_norm_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled
) {
    // Implement CPU version of batch_norm
    // This is a placeholder for the actual implementation
    return at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

torch::Tensor conv_bn_fn(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int64_t stride,
    bool is_training
) {
    std::vector<int64_t> stride_vec = {stride, stride};
    std::vector<int64_t> padding_vec = {1, 1};

    // Convolution
    x = conv2d_cpu(
        x,
        conv_weight,
        stride_vec,
        padding_vec,
        {1, 1},
        1
    );

    // Batch Normalization
    x = batch_norm_cpu(
        x,
        bn_weight,
        bn_bias,
        bn_mean,
        bn_var,
        is_training,
        0.1,
        1e-5,
        true
    );

    // ReLU activation
    x = at::relu(x);
    return x;
}

torch::Tensor conv_dw_fn(
    torch::Tensor x,
    torch::Tensor dw_conv_weight,
    torch::Tensor dw_bn_weight,
    torch::Tensor dw_bn_bias,
    torch::Tensor dw_bn_mean,
    torch::Tensor dw_bn_var,
    torch::Tensor pw_conv_weight,
    torch::Tensor pw_bn_weight,
    torch::Tensor pw_bn_bias,
    torch::Tensor pw_bn_mean,
    torch::Tensor pw_bn_var,
    int64_t stride,
    bool is_training
) {
    // Depthwise Convolution
    std::vector<int64_t> dw_stride_vec = {stride, stride};
    std::vector<int64_t> dw_padding_vec = {1, 1};
    std::vector<int64_t> dw_dilation_vec = {1, 1};
    int64_t groups = dw_conv_weight.size(0);

    x = conv2d_cpu(
        x,
        dw_conv_weight,
        dw_stride_vec,
        dw_padding_vec,
        dw_dilation_vec,
        groups
    );

    x = batch_norm_cpu(
        x,
        dw_bn_weight,
        dw_bn_bias,
        dw_bn_mean,
        dw_bn_var,
        is_training,
        0.1,
        1e-5,
        true
    );

    x = at::relu(x);

    // Pointwise Convolution
    std::vector<int64_t> pw_stride_vec = {1, 1};
    std::vector<int64_t> pw_padding_vec = {0, 0};
    std::vector<int64_t> pw_dilation_vec = {1, 1};

    x = conv2d_cpu(
        x,
        pw_conv_weight,
        pw_stride_vec,
        pw_padding_vec,
        pw_dilation_vec,
        1
    );

    x = batch_norm_cpu(
        x,
        pw_bn_weight,
        pw_bn_bias,
        pw_bn_mean,
        pw_bn_var,
        is_training,
        0.1,
        1e-5,
        true
    );

    x = at::relu(x);
    return x;
}

torch::Tensor forward(
    torch::Tensor x,
    py::object params_obj,
    bool is_training
) {
    // Convert params to a Python dictionary if necessary
    py::dict params = params_obj.cast<py::dict>();

    // First conv+bn+relu
    x = conv_bn_fn(
        x,
        params["conv0_weight"].cast<torch::Tensor>(),
        params["bn0_weight"].cast<torch::Tensor>(),
        params["bn0_bias"].cast<torch::Tensor>(),
        params["bn0_mean"].cast<torch::Tensor>(),
        params["bn0_var"].cast<torch::Tensor>(),
        2,
        is_training
    );

    // 13 conv_dw blocks
    for (int i = 0; i < 13; ++i) {
        std::string idx = std::to_string(i + 1);

        // Depthwise parameters
        std::string conv_dw_weight_key = "conv" + idx + "_dw_weight";
        std::string dw_bn_weight_key = "bn" + idx + "_dw_weight";
        std::string dw_bn_bias_key = "bn" + idx + "_dw_bias";
        std::string dw_bn_mean_key = "bn" + idx + "_dw_mean";
        std::string dw_bn_var_key = "bn" + idx + "_dw_var";

        // Pointwise parameters
        std::string conv_pw_weight_key = "conv" + idx + "_pw_weight";
        std::string pw_bn_weight_key = "bn" + idx + "_pw_weight";
        std::string pw_bn_bias_key = "bn" + idx + "_pw_bias";
        std::string pw_bn_mean_key = "bn" + idx + "_pw_mean";
        std::string pw_bn_var_key = "bn" + idx + "_pw_var";

        int64_t stride = (i == 1 || i == 3 || i == 5 || i == 11) ? 2 : 1;

        x = conv_dw_fn(
            x,
            params[conv_dw_weight_key.c_str()].cast<torch::Tensor>(),
            params[dw_bn_weight_key.c_str()].cast<torch::Tensor>(),
            params[dw_bn_bias_key.c_str()].cast<torch::Tensor>(),
            params[dw_bn_mean_key.c_str()].cast<torch::Tensor>(),
            params[dw_bn_var_key.c_str()].cast<torch::Tensor>(),
            params[conv_pw_weight_key.c_str()].cast<torch::Tensor>(),
            params[pw_bn_weight_key.c_str()].cast<torch::Tensor>(),
            params[pw_bn_bias_key.c_str()].cast<torch::Tensor>(),
            params[pw_bn_mean_key.c_str()].cast<torch::Tensor>(),
            params[pw_bn_var_key.c_str()].cast<torch::Tensor>(),
            stride,
            is_training
        );
    }

    // Average pooling
    std::vector<int64_t> avgpool_kernel = {7, 7};
    x = at::avg_pool2d(x, avgpool_kernel);

    // Flatten and Fully Connected Layer
    x = x.view({x.size(0), -1});
    x = at::linear(
        x,
        params["fc_weight"].cast<torch::Tensor>(),
        params["fc_bias"].cast<torch::Tensor>()
    );

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MobileNetV1 forward pass (CPU)");
}