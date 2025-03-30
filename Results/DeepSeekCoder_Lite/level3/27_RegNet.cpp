#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    std::vector<std::vector<torch::Tensor>> stage_params,
    torch::Tensor fc_weight,
    torch::Tensor fc_bias,
    bool is_training) {

    // Process each stage
    for (auto& params : stage_params) {
        // Unpack parameters for this stage
        auto conv1_weight = params[0];
        auto conv1_bias = params[1];
        auto bn1_weight = params[2];
        auto bn1_bias = params[3];
        auto bn1_mean = params[4];
        auto bn1_var = params[5];
        auto conv2_weight = params[6];
        auto conv2_bias = params[7];
        auto bn2_weight = params[8];
        auto bn2_bias = params[9];
        auto bn2_mean = params[10];
        auto bn2_var = params[11];

        // Conv1 + BN + ReLU
        x = torch::conv2d(x, conv1_weight, conv1_bias, 1, 1);
        x = torch::batch_norm(x, bn1_weight, bn1_bias, bn1_mean, bn1_var, 
                            is_training, 0.1, 1e-5, true);
        x = torch::relu(x);

        // Conv2 + BN + ReLU
        x = torch::conv2d(x, conv2_weight, conv2_bias, 1, 1);
        x = torch::batch_norm(x, bn2_weight, bn2_bias, bn2_mean, bn2_var,
                            is_training, 0.1, 1e-5, true);
        x = torch::relu(x);

        // MaxPool
        x = torch::max_pool2d(x, {2, 2}, {2, 2});
    }

    // Global average pooling
    x = torch::mean(x, {2, 3}, /*keepdim=*/false);

    // Final linear layer
    x = torch::linear(x, fc_weight, fc_bias);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RegNet forward");
}