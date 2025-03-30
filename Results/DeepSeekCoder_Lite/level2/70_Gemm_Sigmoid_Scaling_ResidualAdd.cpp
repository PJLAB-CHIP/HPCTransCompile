#include <torch/extension.h>
#include <vector>
#include <omp.h>

template <typename scalar_t>
void sigmoid_scaling_residual_add_kernel(scalar_t* x_data, const scalar_t* original_x_data, scalar_t scaling_factor, int size)
{
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        scalar_t val = x_data[i];
        scalar_t orig_val = original_x_data[i];
        val = 1.0 / (1.0 + exp(-val));
        val = val * scaling_factor + orig_val;
        x_data[i] = val;
    }
}

void sigmoid_scaling_residual_add(torch::Tensor& x, const torch::Tensor& original_x, float scaling_factor)
{
    const int size = x.numel();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "sigmoid_scaling_residual_add", ([&] {
        sigmoid_scaling_residual_add_kernel<scalar_t>(
            x.data_ptr<scalar_t>(),
            original_x.data_ptr<scalar_t>(),
            static_cast<scalar_t>(scaling_factor),
            size);
    }));
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");

    x = torch::addmm(bias, x, weight.t());
    torch::Tensor original_x = x.clone();
    sigmoid_scaling_residual_add(x, original_x, scaling_factor);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Module function forward (CPU)");
}