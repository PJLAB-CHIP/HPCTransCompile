#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor branch1x1_weight,
    torch::Tensor branch1x1_bias,
    torch::Tensor branch3x3_reduce_weight,
    torch::Tensor branch3x3_reduce_bias,
    torch::Tensor branch3x3_weight,
    torch::Tensor branch3x3_bias,
    torch::Tensor branch5x5_reduce_weight,
    torch::Tensor branch5x5_reduce_bias,
    torch::Tensor branch5x5_weight,
    torch::Tensor branch5x5_bias,
    torch::Tensor branch_pool_conv_weight,
    torch::Tensor branch_pool_conv_bias
) {
    // Make sure all tensors are on the same device
    auto device = x.device();
    branch1x1_weight = branch1x1_weight.to(device);
    branch1x1_bias = branch1x1_bias.to(device);
    branch3x3_reduce_weight = branch3x3_reduce_weight.to(device);
    branch3x3_reduce_bias = branch3x3_reduce_bias.to(device);
    branch3x3_weight = branch3x3_weight.to(device);
    branch3x3_bias = branch3x3_bias.to(device);
    branch5x5_reduce_weight = branch5x5_reduce_weight.to(device);
    branch5x5_reduce_bias = branch5x5_reduce_bias.to(device);
    branch5x5_weight = branch5x5_weight.to(device);
    branch5x5_bias = branch5x5_bias.to(device);
    branch_pool_conv_weight = branch_pool_conv_weight.to(device);
    branch_pool_conv_bias = branch_pool_conv_bias.to(device);

    // 1x1 branch
    auto branch1x1 = at::conv2d(x, branch1x1_weight, branch1x1_bias);

    // 3x3 branch
    auto branch3x3 = at::conv2d(x, branch3x3_reduce_weight, branch3x3_reduce_bias);
    branch3x3 = at::conv2d(branch3x3, branch3x3_weight, branch3x3_bias, 1, 1);

    // 5x5 branch
    auto branch5x5 = at::conv2d(x, branch5x5_reduce_weight, branch5x5_reduce_bias);
    branch5x5 = at::conv2d(branch5x5, branch5x5_weight, branch5x5_bias, 1, 2);

    // Pool branch
    auto branch_pool = at::max_pool2d(x, {3, 3}, {1, 1}, {1, 1});
    branch_pool = at::conv2d(branch_pool, branch_pool_conv_weight, branch_pool_conv_bias);

    // Concatenate outputs along dimension 1
    std::vector<torch::Tensor> outputs = {branch1x1, branch3x3, branch5x5, branch_pool};
    auto output = at::cat(outputs, 1);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Inception module forward (CPU)");
}