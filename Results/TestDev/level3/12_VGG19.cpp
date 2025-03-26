#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor conv_relu_cpu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    x = at::conv2d(x, weight, bias, {1,1}, {1,1});
    return at::relu_(x);
}

torch::Tensor max_pool_cpu(torch::Tensor x) {
    return at::max_pool2d(x, {2,2}, {2,2});
}

torch::Tensor forward_cpu(
    torch::Tensor x,
    std::vector<torch::Tensor> conv_weights,
    std::vector<torch::Tensor> conv_biases,
    std::vector<torch::Tensor> fc_weights,
    std::vector<torch::Tensor> fc_biases,
    bool is_training
) {
    x = x.contiguous().to(torch::kCPU);
    for (auto& w : conv_weights) w = w.contiguous().to(torch::kCPU);

    #pragma omp parallel for
    for (int i = 0; i < conv_weights.size(); ++i) {
        x = conv_relu_cpu(x, conv_weights[i], conv_biases[i]);
    }

    x = max_pool_cpu(x);

    #pragma omp parallel for
    for (int i = 0; i < conv_weights.size(); ++i) {
        x = conv_relu_cpu(x, conv_weights[i], conv_biases[i]);
    }

    x = max_pool_cpu(x);

    #pragma omp parallel for
    for (int i = 0; i < conv_weights.size(); ++i) {
        x = conv_relu_cpu(x, conv_weights[i], conv_biases[i]);
    }

    x = max_pool_cpu(x);

    #pragma omp parallel for
    for (int i = 0; i < conv_weights.size(); ++i) {
        x = conv_relu_cpu(x, conv_weights[i], conv_biases[i]);
    }

    x = max_pool_cpu(x);

    #pragma omp parallel for
    for (int i = 0; i < conv_weights.size(); ++i) {
        x = conv_relu_cpu(x, conv_weights[i], conv_biases[i]);
    }

    x = max_pool_cpu(x).contiguous();

    x = x.flatten(1, -1);
    x = at::linear(x, fc_weights[0], fc_biases[0]).relu_();
    x = at::linear(x, fc_weights[1], fc_biases[1]).relu_();
    x = at::linear(x, fc_weights[2], fc_biases[2]);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "VGG19 forward pass on CPU");
}