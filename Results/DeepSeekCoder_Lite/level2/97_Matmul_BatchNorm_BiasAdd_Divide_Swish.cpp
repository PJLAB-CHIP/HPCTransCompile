#include <torch/extension.h>
#include <vector>
#include <omp.h>

template <typename scalar_t>
void block_tuned_kernel(
    const scalar_t* __restrict__ x_linear,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bn_weight,
    const scalar_t* __restrict__ bn_bias,
    scalar_t* __restrict__ bn_running_mean,
    scalar_t* __restrict__ bn_running_var,
    const scalar_t* __restrict__ add_bias,
    const float bn_eps,
    const float bn_momentum,
    const float divide_value,
    const int batch_size,
    const int out_features) {

    const int threads = 128;
    const int blocks = out_features;

    #pragma omp parallel for
    for (int f = 0; f < out_features; ++f) {
        float s_sum[threads];
        float s_sumsq[threads];

        for (int i = 0; i < threads; ++i) {
            s_sum[i] = 0.0f;
            s_sumsq[i] = 0.0f;
        }

        #pragma omp parallel for
        for (int tid = 0; tid < threads; ++tid) {
            int vec_elements = (batch_size / 4) * 4;
            float thread_sum = 0.0f;
            float thread_sumsq = 0.0f;

            for (int i = tid; i < vec_elements; i += threads) {
                float4 values;
                values.x = static_cast<float>(x_linear[i * out_features + f]);
                values.y = static_cast<float>(x_linear[(i + 1) * out_features + f]);
                values.z = static_cast<float>(x_linear[(i + 2) * out_features + f]);
                values.w = static_cast<float>(x_linear[(i + 3) * out_features + f]);

                thread_sum += values.x + values.y + values.z + values.w;
                thread_sumsq += values.x * values.x + values.y * values.y + values.z * values.z + values.w * values.w;
            }

            for (int i = vec_elements + tid; i < batch_size; i += threads) {
                float val = static_cast<float>(x_linear[i * out_features + f]);
                thread_sum += val;
                thread_sumsq += val * val;
            }

            s_sum[tid] = thread_sum;
            s_sumsq[tid] = thread_sumsq;
        }

        for (int i = 0; i < threads; ++i) {
            for (int j = 1; j < 32; j *= 2) {
                s_sum[i] += s_sum[i + j];
                s_sumsq[i] += s_sumsq[i + j];
            }
        }

        int tid = 0;
        float mean = s_sum[tid] / batch_size;
        float var = (s_sumsq[tid] / batch_size) - (mean * mean);

        bn_running_mean[f] = bn_running_mean[f] * (1 - bn_momentum) + mean * bn_momentum;
        bn_running_var[f] = bn_running_var[f] * (1 - bn_momentum) + var * bn_momentum;

        float inv_std = 1.0f / sqrt(var + bn_eps);
        float gamma = bn_weight[f];
        float beta = bn_bias[f];
        float extra_bias = add_bias[0];

        for (int i = tid; i < batch_size; i += threads) {
            for (int j = 0; j < 4; ++j) {
                int idx = (i * out_features) + f + j;
                float val = static_cast<float>(x_linear[idx]);
                float normalized = (val - mean) * inv_std;
                float transformed = gamma * normalized + beta + extra_bias;
                float divided = transformed / divide_value;
                output[idx] = static_cast<scalar_t>(divided / (1.0f + expf(-divided)));
            }
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

    AT_DISPATCH_FLOATING_TYPES(x_linear.scalar_type(), "block_tuned_kernel", ([&] {
        block_tuned_kernel<scalar_t>(
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