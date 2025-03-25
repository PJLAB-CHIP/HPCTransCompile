#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define ELEMENTS_PER_THREAD 4

float fused_mish_tanh_activation(float x) {
    float softplus = logf(1.0f + expf(x));
    float mish = x * tanhf(softplus);
    return tanhf(mish);
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor x must be a CPU tensor");
    TORCH_CHECK(!conv_weight.is_cuda(), "Convolution weight must be a CPU tensor");
    TORCH_CHECK(!conv_bias.is_cuda(), "Convolution bias must be a CPU tensor");

    auto x_conv = at::conv3d(
        x,
        conv_weight,
        conv_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    auto output = torch::empty_like(x_conv);
    const int total_elements = x_conv.numel();

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i += ELEMENTS_PER_THREAD) {
        for (int j = 0; j < ELEMENTS_PER_THREAD && i + j < total_elements; j++) {
            const float val = x_conv.data_ptr<float>()[i + j];
            const float result = fused_mish_tanh_activation(val);
            output.data_ptr<float>()[i + j] = result;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "CPU optimized convolution with Mish and Tanh activations");
}