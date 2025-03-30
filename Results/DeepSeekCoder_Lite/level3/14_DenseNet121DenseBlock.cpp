#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <omp.h>

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

    if (is_training) {
        // Update running_mean and running_var
        bn_mean = bn_mean * (1 - momentum) + at::mean(x, {0, 2, 3}, true) * momentum;
        bn_var = bn_var * (1 - momentum) + at::var(x, {0, 2, 3}, true, true) * momentum;
    }

    x = at::batch_norm(
        x,
        bn_weight,
        bn_bias,
        bn_mean,
        bn_var,
        is_training,
        momentum,
        eps,
        true
    );

    x = at::relu(x);

    x = at::conv2d(
        x,
        conv_weight,
        {},
        {1, 1},
        {1, 1}
    );

    x = at::dropout(x, 0.0, is_training);

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

    #pragma omp parallel for
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
        x = at::cat(features, 1);
    }

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "DenseNet121 dense block forward function");
}