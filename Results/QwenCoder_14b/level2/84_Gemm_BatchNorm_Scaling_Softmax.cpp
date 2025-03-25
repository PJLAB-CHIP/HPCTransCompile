#include <torch/extension.h>
#include <cmath>
#include <omp.h>

using namespace torch;

void fused_kernel_cpu(
    const float* x,
    const float* gemm_weight,
    const float* gemm_bias,
    const float* running_mean,
    const float* running_var,
    float bn_eps,
    const float* bn_weight,
    const float* bn_bias,
    const float* scale,
    float* output,
    int M, int K, int N
) {
    #pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        float result = gemm_bias[0];
        for (int k = 0; k < K; ++k) {
            result += x[m * K + k] * gemm_weight[k];
        }

        float normalized = (result - running_mean[0]) * std::sqrt(1.0f / (running_var[0] + bn_eps));
        normalized = normalized * bn_weight[0] + bn_bias[0];

        float scaled = normalized * scale[0];

        float max_val = scaled;
        #pragma omp parallel for reduction(max:max_val)
        for (int n = 1; n < N; ++n) {
            float val = scaled;
            if (val > max_val) {
                max_val = val;
            }
        }

        float exp_val = std::exp(scaled - max_val);
        float local_sum = exp_val;

        #pragma omp parallel for reduction(+:local_sum)
        for (int n = 1; n < N; ++n) {
            float val = std::exp(scaled - max_val);
            local_sum += val;
        }

        output[m * N + 0] = exp_val / local_sum;
    }
}

Tensor forward_cpu(
    Tensor x,
    Tensor running_mean,
    Tensor running_var,
    double bn_eps,
    double bn_momentum,
    Tensor bn_weight,
    Tensor bn_bias,
    Tensor scale,
    Tensor gemm_weight,
    Tensor gemm_bias
) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = gemm_bias.size(0);

    auto output = empty({M, N}, x.options());

    fused_kernel_cpu(
        x.data_ptr<float>(),
        gemm_weight.data_ptr<float>(),
        gemm_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(bn_eps),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        M, K, N
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Streamed fused GEMM+BN+Softmax CPU");
}