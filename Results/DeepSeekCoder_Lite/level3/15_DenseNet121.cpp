#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;

at::Tensor dense_layer_fn(
    at::Tensor x,
    at::Tensor bn_weight,
    at::Tensor bn_bias,
    at::Tensor bn_mean,
    at::Tensor bn_var,
    at::Tensor conv_weight,
    bool is_training
) {
    x = at::batch_norm(x, bn_weight, bn_bias, bn_mean, bn_var, is_training, 0.1, 1e-5, true);
    x = at::relu(x);
    x = at::conv2d(x, conv_weight, /*bias=*/{}, /*stride=*/{1, 1}, /*padding=*/{1, 1});
    x = at::dropout(x, /*p=*/0.0, is_training);
    return x;
}

at::Tensor transition_layer_fn(
    at::Tensor x,
    at::Tensor bn_weight,
    at::Tensor bn_bias,
    at::Tensor bn_mean,
    at::Tensor bn_var,
    at::Tensor conv_weight,
    bool is_training
) {
    x = at::batch_norm(x, bn_weight, bn_bias, bn_mean, bn_var, is_training, 0.1, 1e-5, true);
    x = at::relu(x);
    x = at::conv2d(x, conv_weight);
    x = at::avg_pool2d(x, /*kernel_size=*/{2, 2}, /*stride=*/{2, 2});
    return x;
}

at::Tensor module_fn(
    at::Tensor x,
    py::object params,
    bool is_training
) {
    // Helper function to get parameters
    auto get_param = [&](const std::string& key) -> at::Tensor {
        return params.attr("__getitem__")(key.c_str()).cast<at::Tensor>();
    };

    // Initial features
    auto features_conv_weight = get_param("features_conv_weight");
    x = at::conv2d(x, features_conv_weight, /*bias=*/{}, /*stride=*/{2, 2}, /*padding=*/{3, 3});

    auto features_bn_mean = get_param("features_bn_mean");
    auto features_bn_var = get_param("features_bn_var");
    auto features_bn_weight = get_param("features_bn_weight");
    auto features_bn_bias = get_param("features_bn_bias");

    x = at::batch_norm(x, features_bn_weight, features_bn_bias, features_bn_mean, features_bn_var, is_training, 0.1, 1e-5, true);
    x = at::relu(x);
    x = at::max_pool2d(x, /*kernel_size=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1});

    std::vector<int> num_layers = {6, 12, 24, 16};  // Layers per block for DenseNet121

    // Dense blocks and transitions
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
        std::vector<at::Tensor> features;
        features.push_back(x);

        for (int j = 0; j < num_layers[i]; ++j) {
            std::string prefix = "block" + std::to_string(i) + "_layer" + std::to_string(j) + "_";

            auto bn_weight = get_param(prefix + "bn_weight");
            auto bn_bias = get_param(prefix + "bn_bias");
            auto bn_mean = get_param(prefix + "bn_mean");
            auto bn_var = get_param(prefix + "bn_var");
            auto conv_weight = get_param(prefix + "conv_weight");

            at::Tensor new_feature = dense_layer_fn(
                x,
                bn_weight,
                bn_bias,
                bn_mean,
                bn_var,
                conv_weight,
                is_training
            );

            features.push_back(new_feature);

            x = at::cat(features, 1);
        }

        if (i != 3) {  // Apply transition after all blocks except the last
            std::string prefix = "transition" + std::to_string(i) + "_";

            auto bn_weight = get_param(prefix + "bn_weight");
            auto bn_bias = get_param(prefix + "bn_bias");
            auto bn_mean = get_param(prefix + "bn_mean");
            auto bn_var = get_param(prefix + "bn_var");
            auto conv_weight = get_param(prefix + "conv_weight");

            x = transition_layer_fn(
                x,
                bn_weight,
                bn_bias,
                bn_mean,
                bn_var,
                conv_weight,
                is_training
            );
        }
    }

    // Final layers
    auto final_bn_mean = get_param("final_bn_mean");
    auto final_bn_var = get_param("final_bn_var");
    auto final_bn_weight = get_param("final_bn_weight");
    auto final_bn_bias = get_param("final_bn_bias");

    x = at::batch_norm(x, final_bn_weight, final_bn_bias, final_bn_mean, final_bn_var, is_training, 0.1, 1e-5, true);
    x = at::relu(x);
    x = at::adaptive_avg_pool2d(x, {1, 1}).reshape({x.size(0), -1});

    auto classifier_weight = get_param("classifier_weight");
    auto classifier_bias = get_param("classifier_bias");
    x = at::linear(x, classifier_weight, classifier_bias);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "DenseNet121 forward");
}