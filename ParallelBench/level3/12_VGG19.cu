#include <torch/extension.h>
#include <vector>

torch::Tensor conv_relu(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    x = at::conv2d(x, weight, bias, {1,1}, {1,1});
    return at::relu_(x);
}

torch::Tensor max_pool(torch::Tensor x) {
    return at::max_pool2d(x, {2,2}, {2,2});
}

torch::Tensor forward(
    torch::Tensor x,
    std::vector<torch::Tensor> conv_weights,
    std::vector<torch::Tensor> conv_biases,
    std::vector<torch::Tensor> fc_weights,
    std::vector<torch::Tensor> fc_biases,
    bool is_training
) {
    at::globalContext().setBenchmarkCuDNN(true);
    x = x.contiguous().to(torch::MemoryFormat::ChannelsLast);
    for (auto& w : conv_weights) w = w.contiguous().to(torch::MemoryFormat::ChannelsLast);

    x = conv_relu(x, conv_weights[0], conv_biases[0]);
    x = conv_relu(x, conv_weights[1], conv_biases[1]);
    x = max_pool(x);

    x = conv_relu(x, conv_weights[2], conv_biases[2]);
    x = conv_relu(x, conv_weights[3], conv_biases[3]);
    x = max_pool(x);

    x = conv_relu(x, conv_weights[4], conv_biases[4]);
    x = conv_relu(x, conv_weights[5], conv_biases[5]);
    x = conv_relu(x, conv_weights[6], conv_biases[6]);
    x = conv_relu(x, conv_weights[7], conv_biases[7]);
    x = max_pool(x);

    x = conv_relu(x, conv_weights[8], conv_biases[8]);
    x = conv_relu(x, conv_weights[9], conv_biases[9]);
    x = conv_relu(x, conv_weights[10], conv_biases[10]);
    x = conv_relu(x, conv_weights[11], conv_biases[11]);
    x = max_pool(x);

    x = conv_relu(x, conv_weights[12], conv_biases[12]);
    x = conv_relu(x, conv_weights[13], conv_biases[13]);
    x = conv_relu(x, conv_weights[14], conv_biases[14]);
    x = conv_relu(x, conv_weights[15], conv_biases[15]);
    x = max_pool(x).contiguous();

    x = x.flatten(1, -1);
    x = at::linear(x, fc_weights[0], fc_biases[0]).relu_();
    x = at::linear(x, fc_weights[1], fc_biases[1]).relu_();
    x = at::linear(x, fc_weights[2], fc_biases[2]);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "VGG19 forward pass with cuDNN optimizations");
}