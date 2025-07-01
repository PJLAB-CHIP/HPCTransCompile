#include <cstdint>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace at;
namespace py = pybind11;

torch::Tensor conv_bn_fn(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    int64_t stride,
    int64_t padding,
    int64_t groups,
    bool is_training
) {
    std::vector<int64_t> stride_vec = {stride, stride};
    std::vector<int64_t> padding_vec = {padding, padding};
    std::vector<int64_t> pw_dilation_vec = {1, 1};

    // Convolution
    x = at::conv2d(
        x,
        conv_weight,
        /*bias=*/c10::nullopt,
        /*stride=*/stride_vec,
        /*padding=*/padding_vec,
        /*dilation=*/pw_dilation_vec,
        /*groups=*/groups
    );

    // Batch Normalization
    x = at::batch_norm(
        x,
        /*weight=*/bn_weight,
        /*bias=*/bn_bias,
        /*running_mean=*/bn_mean,
        /*running_var=*/bn_var,
        /*training=*/is_training,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/true
    );
    return x;
}


torch::Tensor forward(
    torch::Tensor x,
    py::object params_obj,
    bool is_training,
    bool use_residual
) {
    // Convert params to a Python dictionary if necessary
    py::dict params = params_obj.cast<py::dict>();
    torch:Tensor identity = x;

    if (params.contains("expand_conv_weight")) {
        x = conv_bn_fn(
            x,
            params["expand_conv_weight"].cast<torch::Tensor>(),
            params["expand_conv_bn_weight"].cast<torch::Tensor>(),
            params["expand_conv_bn_bias"].cast<torch::Tensor>(),
            params["expand_conv_bn_running_mean"].cast<torch::Tensor>(),
            params["expand_conv_bn_running_var"].cast<torch::Tensor>(),
            1,
            0,
            1,
            is_training
        );
        x = at::relu6(x);
    }

    int stride = 1;
    if (use_residual) ++stride;
    torch::Tensor weight = params["depthwise_conv_weight"].cast<torch::Tensor>();
    int padding = (weight.size(2) - 1) / 2;
    int groups = weight.size(0);

    x = conv_bn_fn(
        x,
        params["depthwise_conv_weight"].cast<torch::Tensor>(),
        params["depthwise_conv_bn_weight"].cast<torch::Tensor>(),
        params["depthwise_conv_bn_bias"].cast<torch::Tensor>(),
        params["depthwise_conv_bn_running_mean"].cast<torch::Tensor>(),
        params["depthwise_conv_bn_running_var"].cast<torch::Tensor>(),
        stride,
        padding,
        groups,
        is_training
    );
    x = at::relu6(x);

    x = conv_bn_fn(
        x,
        params["project_conv_weight"].cast<torch::Tensor>(),
        params["project_conv_bn_weight"].cast<torch::Tensor>(),
        params["project_conv_bn_bias"].cast<torch::Tensor>(),
        params["project_conv_bn_running_mean"].cast<torch::Tensor>(),
        params["project_conv_bn_running_var"].cast<torch::Tensor>(),
        1,
        0,
        1,
        is_training
    );

    if (use_residual) x += identity;

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "EfficientNetMBConv forward pass (CUDA)");
}