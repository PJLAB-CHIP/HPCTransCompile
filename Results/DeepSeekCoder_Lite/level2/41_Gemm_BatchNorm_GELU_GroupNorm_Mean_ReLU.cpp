#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
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
    auto out = torch::linear(x, gemm_weight, gemm_bias);

    // 2) BatchNorm in training mode
    auto running_mean = batch_norm_running_mean.detach().cpu().numpy();
    auto running_var = batch_norm_running_var.detach().cpu().numpy();
    auto weight = batch_norm_weight.detach().cpu().numpy();
    auto bias = batch_norm_bias.detach().cpu().numpy();
    auto output_np = out.detach().cpu().numpy();

    int64_t N = output_np.shape[0];
    int64_t C = output_np.shape[1];
    int64_t H = output_np.shape[2];
    int64_t W = output_np.shape[3];

    auto output_np_reshaped = output_np.reshape(N * C, -1);

    #pragma omp parallel for
    for (int64_t i = 0; i < N * C; ++i) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int64_t j = 0; j < H * W; ++j) {
            double val = output_np_reshaped[i * H * W + j];
            sum += val;
            sum_sq += val * val;
        }
        double mean = sum / (H * W);
        double var = (sum_sq / (H * W)) - (mean * mean);

        for (int64_t j = 0; j < H * W; ++j) {
            output_np_reshaped[i * H * W + j] = (output_np_reshaped[i * H * W + j] - mean) / std::sqrt(var + 1e-5);
        }

        // Apply batch normalization weight and bias
        for (int64_t j = 0; j < H * W; ++j) {
            output_np_reshaped[i * H * W + j] = output_np_reshaped[i * H * W + j] * weight[i % C] + bias[i % C];
        }
    }

    output_np = output_np_reshaped.reshape(N, C, H, W);

    // 3) GELU
    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < C; ++j) {
            for (int64_t k = 0; k < H * W; ++k) {
                double val = output_np[i][j][k];
                output_np[i][j][k] = 0.5 * val * (1.0 + std::erf(0.7978845608028654 * val));
            }
        }
    }

    // 4) GroupNorm
    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < C; ++j) {
            double sum = 0.0;
            double sum_sq = 0.0;
            for (int64_t k = 0; k < H * W; ++k) {
                double val = output_np[i][j][k];
                sum += val;
                sum_sq += val * val;
            }
            double mean = sum / (H * W);
            double var = (sum_sq / (H * W)) - (mean * mean);

            for (int64_t k = 0; k < H * W; ++k) {
                output_np[i][j][k] = (output_np[i][j][k] - mean) / std::sqrt(var + 1e-5);
            }

            // Apply group normalization weight and bias
            for (int64_t k = 0; k < H * W; ++k) {
                output_np[i][j][k] = output_np[i][j][k] * group_norm_weight[j / (C / num_groups)] + group_norm_bias[j / (C / num_groups)];
            }
        }
    }

    // 5) Mean across dim=1, keepdim=true
    auto output_tensor = torch::from_numpy(output_np);
    output_tensor = output_tensor.mean(1, true);

    // 6) ReLU
    output_tensor = torch::relu(output_tensor);

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward,
        "Fused GEMM-BatchNorm-GELU-GroupNorm-Mean-ReLU forward (CPU)"
    );
}