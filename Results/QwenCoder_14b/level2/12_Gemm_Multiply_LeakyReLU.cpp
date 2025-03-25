#include <torch/extension.h>
#include <omp.h>

float warp_reduce_sum(float val) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += val;
    }
    return val;
}

void module_fn_cpu(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float multiplier,
    const float negative_slope
) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int col = 0; col < out_features; ++col) {
            const float* x_row = x + row * in_features;
            const float* weight_col = weight + col * in_features;
            
            float thread_sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                thread_sum += x_row[k] * weight_col[k];
            }
            
            float sum = warp_reduce_sum(thread_sum);
            
            sum += bias[col];
            sum *= multiplier;
            output[row * out_features + col] = sum > 0 ? sum : sum * negative_slope;
        }
    }
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    float multiplier,
    float negative_slope,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(!x.is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(!weight.is_cuda(), "weight must be a CPU tensor");
    TORCH_CHECK(!bias.is_cuda(), "bias must be a CPU tensor");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "Weight in_features must match x in_features");
    TORCH_CHECK(bias.size(0) == out_features, "Bias size must match weight out_features");

    auto output = torch::zeros({batch_size, out_features}, x.options());

    module_fn_cpu(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        multiplier,
        negative_slope
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Module function forward CPU with warp primitives");
}