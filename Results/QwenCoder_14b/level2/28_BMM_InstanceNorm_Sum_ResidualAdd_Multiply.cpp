#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

template <typename T>
void fused_linear_instancenorm_cpu(
    const T* input,      // [batch_size, in_features]
    const T* residual,   // [batch_size, out_features]
    const T* weight,     // [out_features, in_features]
    const T* bias,       // [out_features]
    T* output,           // [batch_size, out_features]
    const int batch_size,
    const int in_features,
    const int out_features,
    const float eps
) {
    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        std::vector<T> s_linear(out_features, 0);
        std::vector<T> s_scratch(in_features, 0);
        std::vector<T> s_reduction(out_features, 0);

        // Step 1: Compute the linear layer output
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            T partial = 0;
            for (int i = 0; i < in_features; ++i) {
                partial += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
            }
            s_linear[out_idx] = partial + bias[out_idx];
        }

        // Step 2: Compute the mean of the linear outputs
        T mean = 0;
        for (int i = 0; i < out_features; ++i) {
            mean += s_linear[i];
        }
        mean /= out_features;

        // Step 3: Compute the variance
        T var = 0;
        for (int i = 0; i < out_features; ++i) {
            T diff = s_linear[i] - mean;
            var += diff * diff;
        }
        var /= out_features;
        T inv_std = std::sqrt(1.0 / (var + eps));

        // Step 4: Normalize the linear output and apply residual addition and multiplication
        for (int i = 0; i < out_features; ++i) {
            T norm_val = (s_linear[i] - mean) * inv_std;
            T res_val = residual[batch_idx * out_features + i];
            output[batch_idx * out_features + i] = (norm_val + res_val) * res_val;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor y,
    float eps,
    float momentum,  // For API compatibility
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be on CPU device");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = y.size(1);

    auto output = torch::empty_like(y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_linear_instancenorm_cpu", ([&] {
        fused_linear_instancenorm_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused linear, instance norm, residual add and multiply on CPU");
}