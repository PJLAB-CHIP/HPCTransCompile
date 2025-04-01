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

torch::Tensor inverted_residual_fn(
    torch::Tensor x,
    py::dict params,
    std::string block_idx,
    bool is_training
) {
    std::string prefix = "block" + block_idx + "_";
    std::string weight_key;
    std::string bn_weight_key;
    std::string bn_bias_key;
    std::string bn_mean_key;
    std::string bn_var_key;
    std::string stride_key;

    if (block_idx != "0") {
        weight_key = prefix + "conv1_weight";
        bn_weight_key = prefix + "bn1_weight";
        bn_bias_key = prefix + "bn1_bias";
        bn_mean_key = prefix + "bn1_running_mean";
        bn_var_key = prefix + "bn1_running_var";

        x = conv_bn_fn(
            x,
            params[weight_key.c_str()].cast<torch::Tensor>(),
            params[bn_weight_key.c_str()].cast<torch::Tensor>(),
            params[bn_bias_key.c_str()].cast<torch::Tensor>(),
            params[bn_mean_key.c_str()].cast<torch::Tensor>(),
            params[bn_var_key.c_str()].cast<torch::Tensor>(),
            1,
            0,
            1,
            is_training
        );
        x = at::relu6(x);
    };

    weight_key = prefix + "conv2_weight";
    bn_weight_key = prefix + "bn2_weight";
    bn_bias_key = prefix + "bn2_bias";
    bn_mean_key = prefix + "bn2_running_mean";
    bn_var_key = prefix + "bn2_running_var";
    stride_key = prefix + "stride";
    torch::Tensor s = params[stride_key.c_str()].cast<torch::Tensor>();
    int stride = s.item<int>();
    auto weight = params[weight_key.c_str()].cast<torch::Tensor>();
    int groups = weight.size(0);

    x = conv_bn_fn(
        x,
        weight,
        params[bn_weight_key.c_str()].cast<torch::Tensor>(),
        params[bn_bias_key.c_str()].cast<torch::Tensor>(),
        params[bn_mean_key.c_str()].cast<torch::Tensor>(),
        params[bn_var_key.c_str()].cast<torch::Tensor>(),
        stride,
        1,
        groups,
        is_training
    );
    x = at::relu6(x);

    weight_key = prefix + "conv3_weight";
    bn_weight_key = prefix + "bn3_weight";
    bn_bias_key = prefix + "bn3_bias";
    bn_mean_key = prefix + "bn3_running_mean";
    bn_var_key = prefix + "bn3_running_var";

    x = conv_bn_fn(
        x,
        params[weight_key.c_str()].cast<torch::Tensor>(),
        params[bn_weight_key.c_str()].cast<torch::Tensor>(),
        params[bn_bias_key.c_str()].cast<torch::Tensor>(),
        params[bn_mean_key.c_str()].cast<torch::Tensor>(),
        params[bn_var_key.c_str()].cast<torch::Tensor>(),
        1,
        0,
        1,
        is_training
    );

    std::string res_key = prefix + "residual";
    if (params.contains(res_key)) {
        x = x + params[res_key.c_str()].cast<torch::Tensor>();
    }

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
        params["bn0_running_mean"].cast<torch::Tensor>(),
        params["bn0_running_var"].cast<torch::Tensor>(),
        2,
        1,
        1,
        is_training
    );
    x = at::relu6(x);

    // 13 conv_dw blocks
    for (int i = 0; i < 17; ++i) {
        std::string idx = std::to_string(i);

        x = inverted_residual_fn(
            x,
            params,
            idx,
            is_training
        );
    }

    x = conv_bn_fn(
        x,
        params["conv_last_weight"].cast<torch::Tensor>(),
        params["bn_last_weight"].cast<torch::Tensor>(),
        params["bn_last_bias"].cast<torch::Tensor>(),
        params["bn_last_running_mean"].cast<torch::Tensor>(),
        params["bn_last_running_var"].cast<torch::Tensor>(),
        1,
        0,
        1,
        is_training
    );
    x = at::relu6(x);

    // Average pooling
    x = at::adaptive_avg_pool2d(x, {1, 1});

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
    m.def("forward", &forward, "MobileNetV2 forward pass (CUDA)");
}