#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>

template <typename scalar_t>
void optimized_hybrid_conv3d_kernel(
    const scalar_t* __restrict__ output,
    const scalar_t* __restrict__ scaling_factor,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ result,
    const int batch_size,
    const int out_channels,
    const int depth,
    const int height,
    const int width) {

    const int total_elements = batch_size * out_channels * depth * height * width;
    const int whd = width * height * depth;

    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; ++idx) {
        int c_idx = (idx / whd) % out_channels;
        scalar_t val = output[idx];
        val *= scaling_factor[c_idx];
        val = tanh(val);
        val *= bias[c_idx];
        val = 1.0 / (1.0 + exp(-val));
        result[idx] = val;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor scaling_factor,
    torch::Tensor bias) {

    auto conv_out = torch::conv3d(x, conv_weight, conv_bias);

    const int batch_size = conv_out.size(0);
    const int out_channels = conv_out.size(1);
    const int depth = conv_out.size(2);
    const int height = conv_out.size(3);
    const int width = conv_out.size(4);

    auto result = torch::empty_like(conv_out);

    const int total_elements = batch_size * out_channels * depth * height * width;

    optimized_hybrid_conv3d_kernel<scalar_t>(
        conv_out.data_ptr<scalar_t>(),
        scaling_factor.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        result.data_ptr<scalar_t>(),
        batch_size,
        out_channels,
        depth,
        height,
        width
    );

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized hybrid Conv3d forward");
}