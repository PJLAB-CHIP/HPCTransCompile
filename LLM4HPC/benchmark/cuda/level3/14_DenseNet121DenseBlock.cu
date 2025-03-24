#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

torch::Tensor layer_fn(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor conv_weight,
    bool is_training
) {
    const double momentum = 0.1;
    const double eps = 1e-5;

    x = at::batch_norm(
        x,
        /*weight=*/bn_weight,
        /*bias=*/bn_bias,
        /*running_mean=*/bn_mean,
        /*running_var=*/bn_var,
        /*training=*/is_training,
        /*momentum=*/momentum,
        /*eps=*/eps,
        /*cudnn_enabled=*/true
    );

    x = at::relu(x);

    x = at::conv2d(
        /*input=*/x,
        /*weight=*/conv_weight,
        /*bias=*/{},
        /*stride=*/{1,1},
        /*padding=*/{1,1}
    );

    x = at::dropout(x, /*p=*/0.0, /*training=*/is_training);

    return x;
}

torch::Tensor forward(
    torch::Tensor x,
    py::object params,
    bool is_training
) {
    // Access the lists from the ParameterDict
    py::list bn_weights = params.attr("__getitem__")("bn_weights");
    py::list bn_biases = params.attr("__getitem__")("bn_biases");
    py::list bn_means = params.attr("__getitem__")("bn_means");
    py::list bn_vars = params.attr("__getitem__")("bn_vars");
    py::list conv_weights = params.attr("__getitem__")("conv_weights");

    std::vector<torch::Tensor> features;
    features.push_back(x);

    size_t num_layers = bn_weights.size();

    for (size_t i = 0; i < num_layers; ++i) {
        torch::Tensor bn_weight = bn_weights[i].cast<torch::Tensor>();
        torch::Tensor bn_bias = bn_biases[i].cast<torch::Tensor>();
        torch::Tensor bn_mean = bn_means[i].cast<torch::Tensor>();
        torch::Tensor bn_var = bn_vars[i].cast<torch::Tensor>();
        torch::Tensor conv_weight = conv_weights[i].cast<torch::Tensor>();

        torch::Tensor new_feature = layer_fn(
            x,
            bn_weight,
            bn_bias,
            bn_mean,
            bn_var,
            conv_weight,
            is_training
        );

        features.push_back(new_feature);
        x = at::cat(features, /*dim=*/1);
    }

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "DenseNet121 dense block forward function");
}