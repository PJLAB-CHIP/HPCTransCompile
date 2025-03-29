#include <torch/extension.h>
#include <omp.h>
#include <cmath>

template <typename scalar_t>
void layernorm_cpu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int outer_size) {

    #pragma omp parallel for collapse(2)
    for (int instance_idx = 0; instance_idx < outer_size; ++instance_idx) {
        const scalar_t* in_ptr = input + instance_idx * normalized_size;
        scalar_t* out_ptr = output + instance_idx * normalized_size;

        scalar_t local_sum = 0;
        scalar_t local_sum_sq = 0;

        for (int idx = 0; idx < normalized_size; ++idx) {
            scalar_t val = in_ptr[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }

        scalar_t mean = local_sum / normalized_size;
        scalar_t variance = (local_sum_sq / normalized_size) - (mean * mean);
        scalar_t inv_std = std::sqrt(1.0f / (variance + eps));

        for (int idx = 0; idx < normalized_size; ++idx) {
            scalar_t val = in_ptr[idx];
            scalar_t normalized = (val - mean) * inv_std;
            out_ptr[idx] = normalized * weight[idx] + bias[idx];
        }
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);

    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;

    layernorm_cpu_kernel<torch::scalar_t>(x.data_ptr<torch::scalar_t>(), weight.data_ptr<torch::scalar_t>(), bias.data_ptr<torch::scalar_t>(), static_cast<float>(eps), output.data_ptr<torch::scalar_t>(), normalized_size, outer_size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CPU)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
