#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// CPU forward declarations
torch::Tensor module_fn_cpu_forward(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    torch::Tensor multiply_weight,
    int64_t num_groups
);

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

template<typename scalar_t>
void module_fn_cpu_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ group_norm_weight,
    const scalar_t* __restrict__ group_norm_bias,
    const scalar_t* __restrict__ multiply_weight,
    const int C,
    const int channels_per_group,
    const int chunk_size
) {
    #pragma omp parallel for collapse(2)
    for (int g = 0; g < num_groups; ++g) {
        for (int n = 0; n < chunk_size; ++n) {
            scalar_t sum = 0.0f;
            scalar_t sumsq = 0.0f;

            for (int c = 0; c < channels_per_group; ++c) {
                const int channel_idx = g * channels_per_group + c;
                const int idx = n * C + channel_idx;
                scalar_t val = x[idx];
                sum += val;
                sumsq += val * val;
            }

            scalar_t mean = sum / channels_per_group;
            scalar_t var = sumsq / channels_per_group - mean * mean + 1e-5f;
            scalar_t inv_std = 1.0f / sqrtf(var);

            for (int c = 0; c < channels_per_group; ++c) {
                const int channel_idx = g * channels_per_group + c;
                const int idx = n * C + channel_idx;
                
                scalar_t val = x[idx];
                scalar_t gamma = group_norm_weight[channel_idx];
                scalar_t beta = group_norm_bias[channel_idx];
                scalar_t w = multiply_weight[channel_idx];

                scalar_t y = (val - mean) * inv_std;
                y = gamma * y + beta;

                scalar_t sigmoid_y = 1.0f / (1.0f + expf(-y));
                y = y * sigmoid_y;

                y = y * w;

                sigmoid_y = 1.0f / (1.0f + expf(-y));
                y = y * sigmoid_y;

                output[idx] = y;
            }
        }
    }
}

torch::Tensor module_fn_cpu_forward(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    torch::Tensor multiply_weight,
    int64_t num_groups
) {
    CHECK_INPUT(x);
    CHECK_INPUT(gemm_weight);
    CHECK_INPUT(gemm_bias);
    CHECK_INPUT(group_norm_weight);
    CHECK_INPUT(group_norm_bias);
    CHECK_INPUT(multiply_weight);

    auto x_linear = torch::addmm(gemm_bias, x, gemm_weight.t());
    auto output = torch::empty_like(x_linear);

    auto N = x_linear.size(0);
    auto C = x_linear.size(1);
    int channels_per_group = C / num_groups;
    
    const int chunk_size = (N + num_groups - 1) / num_groups;
    
    module_fn_cpu_kernel<scalar_t>(
        x_linear.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        group_norm_weight.data_ptr<scalar_t>(),
        group_norm_bias.data_ptr<scalar_t>(),
        multiply_weight.data_ptr<scalar_t>(),
        C,
        channels_per_group,
        chunk_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu_forward, "Module function forward");
}