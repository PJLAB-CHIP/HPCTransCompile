#include <torch/extension.h>
#include <cmath>
#include <omp.h>

float softplus(float x) {
    float abs_x = std::fabs(x);
    float z = std::exp(-abs_x);
    return std::max(x, 0.0f) + std::log1p(z);
}

float mish(float x) {
    float sp = softplus(x);
    return x * std::tanh(sp);
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "bias shape mismatch");

    auto output = torch::empty({batch_size, out_features}, x.options());

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += x[i * in_features + k] * weight[j * in_features + k];
            }
            sum += bias[j];

            float y = mish(sum);
            y = mish(y);
            output[i * out_features + j] = y;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Linear double Mish forward (CPU)");
}