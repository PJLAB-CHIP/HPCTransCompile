#include <pybind11/pybind11.h>
#include <torch/extension.h>

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
    auto out = torch::linear(x, gemm_weight, gemm_bias);

    // 2) BatchNorm in training mode
    out = torch::batch_norm(
        out,
        batch_norm_running_mean,
        batch_norm_running_var,
        batch_norm_weight,
        batch_norm_bias,
        /*training=*/true,
        /*momentum=*/0.1,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/true
    );

    // 3) GELU
    out = torch::gelu(out);

    // 4) GroupNorm
    out = torch::group_norm(
        out,
        num_groups,
        group_norm_weight,
        group_norm_bias,
        /*eps=*/1e-5
    );

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
        "Fused GEMM-BatchNorm-GELU-GroupNorm-Mean-ReLU forward (CUDA)"
    );
}