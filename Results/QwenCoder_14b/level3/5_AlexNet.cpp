#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>

namespace py = pybind11;

torch::Tensor forward(torch::Tensor x, py::object params, bool is_training) {
    // Extract parameters using Python API for ParameterDict compatibility
    torch::Tensor conv1_weight = params.attr("get")("conv1_weight").cast<torch::Tensor>();
    torch::Tensor conv1_bias = params.attr("get")("conv1_bias").cast<torch::Tensor>();
    torch::Tensor conv2_weight = params.attr("get")("conv2_weight").cast<torch::Tensor>();
    torch::Tensor conv2_bias = params.attr("get")("conv2_bias").cast<torch::Tensor>();
    torch::Tensor conv3_weight = params.attr("get")("conv3_weight").cast<torch::Tensor>();
    torch::Tensor conv3_bias = params.attr("get")("conv3_bias").cast<torch::Tensor>();
    torch::Tensor conv4_weight = params.attr("get")("conv4_weight").cast<torch::Tensor>();
    torch::Tensor conv4_bias = params.attr("get")("conv4_bias").cast<torch::Tensor>();
    torch::Tensor conv5_weight = params.attr("get")("conv5_weight").cast<torch::Tensor>();
    torch::Tensor conv5_bias = params.attr("get")("conv5_bias").cast<torch::Tensor>();
    torch::Tensor fc1_weight = params.attr("get")("fc1_weight").cast<torch::Tensor>();
    torch::Tensor fc1_bias = params.attr("get")("fc1_bias").cast<torch::Tensor>();
    torch::Tensor fc2_weight = params.attr("get")("fc2_weight").cast<torch::Tensor>();
    torch::Tensor fc2_bias = params.attr("get")("fc2_bias").cast<torch::Tensor>();
    torch::Tensor fc3_weight = params.attr("get")("fc3_weight").cast<torch::Tensor>();
    torch::Tensor fc3_bias = params.attr("get")("fc3_bias").cast<torch::Tensor>();

    // Use OpenMP for parallelization
    #pragma omp parallel for
    for (int i = 0; i < x.size(0); ++i) {
        x[i] = torch::conv2d(x[i], conv1_weight, conv1_bias, {4, 4}, {2, 2});
        x[i] = torch::relu(x[i]);
        x[i] = torch::max_pool2d(x[i], {3, 3}, {2, 2});

        x[i] = torch::conv2d(x[i], conv2_weight, conv2_bias, {1, 1}, {2, 2});
        x[i] = torch::relu(x[i]);
        x[i] = torch::max_pool2d(x[i], {3, 3}, {2, 2});

        x[i] = torch::conv2d(x[i], conv3_weight, conv3_bias, {1, 1}, {1, 1});
        x[i] = torch::relu(x[i]);

        x[i] = torch::conv2d(x[i], conv4_weight, conv4_bias, {1, 1}, {1, 1});
        x[i] = torch::relu(x[i]);

        x[i] = torch::conv2d(x[i], conv5_weight, conv5_bias, {1, 1}, {1, 1});
        x[i] = torch::relu(x[i]);
        x[i] = torch::max_pool2d(x[i], {3, 3}, {2, 2});

        x[i] = x[i].flatten(1);

        x[i] = torch::linear(x[i], fc1_weight, fc1_bias);
        x[i] = torch::relu(x[i]);
        x[i] = torch::dropout(x[i], 0.0, is_training);

        x[i] = torch::linear(x[i], fc2_weight, fc2_bias);
        x[i] = torch::relu(x[i]);
        x[i] = torch::dropout(x[i], 0.0, is_training);

        x[i] = torch::linear(x[i], fc3_weight, fc3_bias);
    }

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "AlexNet forward (ParameterDict-compatible)");
}