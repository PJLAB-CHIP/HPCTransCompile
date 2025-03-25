#include <torch/extension.h>
#include <omp.h>
#include <cmath>

template <typename scalar_t>
void block_tuned_cpu(
    const scalar_t* x_linear,
    scalar_t* output,
    const scalar_t* bn_weight,
    const scalar_t* bn_bias,
    scalar_t* bn_running_mean,
    scalar_t* bn_running_var,
    const scalar_t* add_bias,
    const float bn_eps,
    const float bn_momentum,
    const float divide_value,
    const int batch_size,
    const int out_features) {

    #pragma omp parallel for
    for (int f = 0; f < out_features; ++f) {
        double local_sum = 0.0;
        double local_sumsq = 0.0;

        const int vec_size = 4;
        const int vec_elements = (batch_size / vec_size) * vec_size;

        #pragma omp parallel for reduction(+:local_sum, local_sumsq)
        for (int i = 0; i < vec_elements; i += vec_size) {
            float4 values;
            values.x = static_cast<float>(x_linear[i * out_features + f]);
            values.y = static_cast<float>(x_linear[(i + 1) * out_features + f]);
            values.z = static_cast<float>(x_linear[(i + 2) * out_features + f]);
            values.w = static_cast<float>(x_linear[(i + 3) * out_features + f]);

            local_sum += values.x + values.y + values.z + values.w;
            local_sumsq += values.x * values.x + values.y * values.y + values.z * values.z + values.w * values.w;
        }

        for (int i = vec_elements; i < batch_size; ++i) {
            float val = static_cast<float>(x_linear[i * out_features + f]);
            local_sum += val;
            local_sumsq += val * val;
        }

        double mean = local_sum / batch_size;
        double var = (local_sumsq / batch_size) - (mean * mean);

        bn_running_mean[f] = bn_running_mean[f] * (1 - bn_momentum) + mean * bn_momentum;
        bn_running_var[f] = bn_running_var[f] * (1 - bn_momentum) + var * bn_momentum;

        const float inv_std = std::rsqrt(var + bn_eps);
        const float gamma = bn_weight[f];
        const float beta = bn_bias[f];
        const float extra_bias = add_bias[0];

        #pragma omp parallel for
        for (int i = 0; i < batch_size; ++i) {
            const int idx = i * out_features + f;
            float val = static_cast<float>(x_linear[idx]);
            float normalized = (val - mean) * inv_std;
            float transformed = fmaf(normalized, gamma, beta) + extra_bias;
            float divided = transformed / divide_value;
            output[idx] = static_cast<scalar_t>(divided / (1.0f + expf(-divided)));
        }
    }
}

torch::Tensor module_fn_cpu(
    torch::Tensor x,
    float bn_eps,
    float bn_momentum,
    float divide_value,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor add_bias) {

    const auto batch_size = x.size(0);
    const auto out_features = weight.size(0);

    auto x_linear = torch::addmm(bias, x, weight.t());
    auto output = torch::empty_like(x_linear);

    AT_DISPATCH_FLOATING_TYPES(x_linear.scalar_type(), "block_tuned_cpu", ([&] {
        block_tuned_cpu<scalar_t>(
            x_linear.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            bn_weight.data_ptr<scalar_t>(),
            bn_bias.data_ptr<scalar_t>(),
            bn_running_mean.data_ptr<scalar_t>(),
            bn_running_var.data_ptr<scalar_t>(),
            add_bias.data_ptr<scalar_t>(),
            bn_eps,
            bn_momentum,
            divide_value,
            batch_size,
            out_features);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu, "Block tuned forward (CPU)");
}