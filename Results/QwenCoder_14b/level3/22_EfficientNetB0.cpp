#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>
#include <omp.h>

namespace py = pybind11;

torch::Tensor forward(
    torch::Tensor x,
    py::object params_obj,  // Accept generic Python object
    bool is_training) {

    // Convert ParameterDict to regular dict if needed
    py::dict params = py::dict(params_obj.attr("items")());

    // Initial conv
    auto conv1_weight = params["conv1_weight"].cast<torch::Tensor>();
    x = torch::conv2d(x, conv1_weight, {}, 2, 1);
    x = torch::batch_norm(
        x,
        params["bn1_weight"].cast<torch::Tensor>(),
        params["bn1_bias"].cast<torch::Tensor>(),
        params["bn1_running_mean"].cast<torch::Tensor>(),
        params["bn1_running_var"].cast<torch::Tensor>(),
        is_training,
        0.9,
        1e-5,
        true
    );
    x = torch::relu(x);

    // MBConv blocks
    std::vector<std::pair<int, int>> block_configs = {
        {1, 1}, {6, 2}, {6, 1}, {6, 2}, {6, 1}, {6, 2}, {6, 1},
        {6, 1}, {6, 1}, {6, 2}, {6, 1}, {6, 1}, {6, 1}
    };

    #pragma omp parallel for
    for (int i = 0; i < block_configs.size(); ++i) {
        int expand_ratio = block_configs[i].first;
        int stride = block_configs[i].second;

        // Convert nested ParameterDict to regular dict
        std::string block_key = "block" + std::to_string(i);
        py::dict block_params = py::dict(
            params[py::str(block_key)].attr("items")()
        );

        auto project_conv_weight = block_params["project_conv_weight"].cast<torch::Tensor>();
        bool use_residual = (stride == 1) && (x.size(1) == project_conv_weight.size(0));

        torch::Tensor identity = x.clone();
        int hidden_dim = x.size(1) * expand_ratio;

        if (expand_ratio != 1) {
            auto expand_conv_weight = block_params["expand_conv_weight"].cast<torch::Tensor>();
            x = torch::conv2d(x, expand_conv_weight, {});
            x = torch::batch_norm(
                x,
                block_params["expand_conv_bn_weight"].cast<torch::Tensor>(),
                block_params["expand_conv_bn_bias"].cast<torch::Tensor>(),
                block_params["expand_conv_bn_running_mean"].cast<torch::Tensor>(),
                block_params["expand_conv_bn_running_var"].cast<torch::Tensor>(),
                is_training,
                0.9,
                1e-5,
                true
            );
            x = torch::clamp(x, 0, 6);
        }

        auto depthwise_conv_weight = block_params["depthwise_conv_weight"].cast<torch::Tensor>();
        int padding = (depthwise_conv_weight.size(2) - 1) / 2;
        x = torch::conv2d(
            x,
            depthwise_conv_weight,
            {},
            stride,
            padding,
            1,
            hidden_dim
        );
        x = torch::batch_norm(
            x,
            block_params["depthwise_conv_bn_weight"].cast<torch::Tensor>(),
            block_params["depthwise_conv_bn_bias"].cast<torch::Tensor>(),
            block_params["depthwise_conv_bn_running_mean"].cast<torch::Tensor>(),
            block_params["depthwise_conv_bn_running_var"].cast<torch::Tensor>(),
            is_training,
            0.9,
            1e-5,
            true
        );
        x = torch::clamp(x, 0, 6);

        x = torch::conv2d(x, project_conv_weight, {});
        x = torch::batch_norm(
            x,
            block_params["project_conv_bn_weight"].cast<torch::Tensor>(),
            block_params["project_conv_bn_bias"].cast<torch::Tensor>(),
            block_params["project_conv_bn_running_mean"].cast<torch::Tensor>(),
            block_params["project_conv_bn_running_var"].cast<torch::Tensor>(),
            is_training,
            0.9,
            1e-5,
            true
        );

        if (use_residual) {
            x += identity;
        }
    }

    // Final conv
    auto conv2_weight = params["conv2_weight"].cast<torch::Tensor>();
    x = torch::conv2d(x, conv2_weight, {});
    x = torch::batch_norm(
        x,
        params["bn2_weight"].cast<torch::Tensor>(),
        params["bn2_bias"].cast<torch::Tensor>(),
        params["bn2_running_mean"].cast<torch::Tensor>(),
        params["bn2_running_var"].cast<torch::Tensor>(),
        is_training,
        0.9,
        1e-5,
        true
    );
    x = torch::relu(x);

    // Final layers
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    x = x.view({x.size(0), -1});
    x = torch::linear(
        x,
        params["fc_weight"].cast<torch::Tensor>(),
        params["fc_bias"].cast<torch::Tensor>()
    );

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "EfficientNetB0 forward");
}