#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <omp.h>

namespace py = pybind11;

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& gemm_weight,
    const torch::Tensor& gemm_bias,
    const torch::Tensor& batch_norm_weight,
    const torch::Tensor& batch_norm_bias,
    const torch::Tensor& batch_norm_running_mean,
    const torch::Tensor& batch_norm_running_var,
    const torch::Tensor& group_norm_weight,
    const torch::Tensor& group_norm_bias,
    const int64_t num_groups
) {
    // 1) GEMM (linear layer)
    auto out = torch::matmul(x, gemm_weight.t()) + gemm_bias;

    // 2) BatchNorm in training mode
    auto mean = out.mean(/*dim=*/1, /*keepdim=*/true);
    auto var = out.var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/true);
    auto normalized = (out - mean) / torch::sqrt(var + 1e-5);
    out = normalized * batch_norm_weight + batch_norm_bias;

    // 3) GELU
    out = 0.5 * out * (1 + torch::erf(out / std::sqrt(2)));

    // 4) GroupNorm
    auto N = out.size(0);
    auto C = out.size(1);
    auto H = out.size(2);
    auto W = out.size(3);
    auto group_size = C / num_groups;
    auto out_reshaped = out.view({N, num_groups, group_size, H, W});
    auto mean_group = out_reshaped.mean(/*dim=*/{2, 3, 4}, /*keepdim=*/true);
    auto var_group = out_reshaped.var(/*dim=*/{2, 3, 4}, /*unbiased=*/false, /*keepdim=*/true);
    auto normalized_group = (out_reshaped - mean_group) / torch::sqrt(var_group + 1e-5);
    out = normalized_group.view({N, C, H, W}) * group_norm_weight + group_norm_bias;

    // 5) Mean across dim=1, keepdim=true
    out = out.mean(/*dim=*/1, /*keepdim=*/true);

    // 6) ReLU
    out = torch::relu(out);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        "Fused GEMM-BatchNorm-GELU-GroupNorm-Mean-ReLU forward (CPU)"
    );
}