#include <torch/extension.h>
#include <vector>
#include <omp.h>

void module_fn_forward_cpu(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float scaling_factor)
{
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            float val = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                val += x[row * in_features + k] * weight[col * in_features + k];
            }
            val += bias[col];
            float original_val = val;
            val *= scaling_factor;
            val += original_val;
            out[row * out_features + col] = val;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    const float scaling_factor,
    torch::Tensor weight,
    torch::Tensor bias)
{
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(!weight.is_cuda(), "weight must be a CPU tensor");
    TORCH_CHECK(!bias.is_cuda(), "bias must be a CPU tensor");

    auto x_ = x.contiguous();
    auto w_ = weight.contiguous();
    auto b_ = bias.contiguous();

    const int batch_size = x_.size(0);
    const int in_features = x_.size(1);
    const int out_features = w_.size(0);

    auto out = torch::empty({batch_size, out_features}, x_.options());

    module_fn_forward_cpu(
        x_.data_ptr<float>(),
        w_.data_ptr<float>(),
        b_.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CPU)");
}