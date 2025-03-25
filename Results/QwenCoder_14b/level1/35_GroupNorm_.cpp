#include <torch/extension.h>
#include <omp.h>
#include <cmath>

typedef float4 float4_t;

// Function to compute per-group mean and variance
void compute_stats_cpu(
    const float* x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    float* mean,
    float* var) {

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            const int group_offset = n * C * spatial + g * channels_per_group * spatial;
            const int group_elems = channels_per_group * spatial;
            
            float thread_sum = 0;
            float thread_sum_sq = 0;

            const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
            const int num_vectors = group_elems / 4;
            const int remaining = group_elems % 4;

            #pragma omp parallel for reduction(+:thread_sum, thread_sum_sq)
            for (int i = 0; i < num_vectors; ++i) {
                float4_t v = x_vec[i];
                thread_sum += v.x + v.y + v.z + v.w;
                thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }

            for (int i = 0; i < remaining; ++i) {
                const float val = x[group_offset + num_vectors * 4 + i];
                thread_sum += val;
                thread_sum_sq += val * val;
            }

            const float group_mean = thread_sum / group_elems;
            const float group_var = thread_sum_sq / group_elems - group_mean * group_mean;
            const int out_index = n * num_groups + g;
            mean[out_index] = group_mean;
            var[out_index] = group_var;
        }
    }
}

// Function to apply the group normalization
void group_norm_forward_cpu(
    const float* x,
    const float* mean,
    const float* var,
    const float* weight,
    const float* bias,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const float eps,
    float* y) {

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            const int group_offset = n * C * spatial + g * channels_per_group * spatial;
            const int group_elems = channels_per_group * spatial;

            for (int i = 0; i < group_elems; ++i) {
                const int idx = group_offset + i;
                const int j = idx % spatial;
                const int temp = idx / spatial;
                const int c = temp % C;
                const int stats_index = n * num_groups + g;

                const float m = mean[stats_index];
                const float v = var[stats_index];
                const float inv_std = std::rsqrt(v + eps);
                const float w = weight[c];
                const float b = bias[c];

                y[idx] = ((x[idx] - m) * inv_std * w + b);
            }
        }
    }
}

// Host function to launch the optimized kernels with OpenMP
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

    const int N = x.size(0);
    const int C = x.size(1);
    int spatial = 1;
    for (int i = 2; i < x.dim(); i++) {
        spatial *= x.size(i);
    }
    const int channels_per_group = C / num_groups;

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);

    compute_stats_cpu(
        x.data_ptr<float>(),
        N, C, spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<float>(),
        var.data_ptr<float>());

    group_norm_forward_cpu(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, spatial,
        channels_per_group,
        num_groups,
        static_cast<float>(eps),
        y.data_ptr<float>());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_forward, "Group Normalization forward (CPU)");
}