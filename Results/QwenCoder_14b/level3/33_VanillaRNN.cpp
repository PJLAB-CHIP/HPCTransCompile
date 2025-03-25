#include <torch/extension.h>
#include <cmath>
#include <omp.h>

void fused_concat_linear_cpu(
    const float* x,
    const float* hidden,
    const float* i2h_weight,
    const float* i2h_bias,
    float* hidden_new_out,
    const int batch_size,
    const int x_size,
    const int hidden_size,
    const int out_size
) {
    int total_width = x_size + hidden_size;

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int out_idx = 0; out_idx < out_size; ++out_idx) {
            float sum = 0.0f;
            for (int k = 0; k < total_width; ++k) {
                float a = (k < x_size) ? x[row * x_size + k] : hidden[row * hidden_size + (k - x_size)];
                float b = i2h_weight[out_idx * total_width + k];
                sum += a * b;
            }
            float result = tanhf(sum + i2h_bias[out_idx]);
            hidden_new_out[row * out_size + out_idx] = result;
        }
    }
}

torch::Tensor module_fn_cpu(
    torch::Tensor x,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias,
    torch::Tensor hidden
) {
    x = x.contiguous();
    i2h_weight = i2h_weight.contiguous();
    i2h_bias = i2h_bias.contiguous();
    h2o_weight = h2o_weight.contiguous();
    h2o_bias = h2o_bias.contiguous();
    hidden = hidden.contiguous();

    const int batch_size = x.size(0);
    const int x_size = x.size(1);
    const int hidden_size = hidden.size(1);
    const int out_size = i2h_bias.size(0);
    int total_width = x_size + hidden_size;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor hidden_new = torch::empty({batch_size, out_size}, options);

    fused_concat_linear_cpu(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        i2h_weight.data_ptr<float>(),
        i2h_bias.data_ptr<float>(),
        hidden_new.data_ptr<float>(),
        batch_size,
        x_size,
        hidden_size,
        out_size
    );

    torch::Tensor output = torch::addmm(h2o_bias, hidden_new, h2o_weight.t());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu, "Fused Module forward (CPU)");
}