#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>

// Tiling parameter for the channel dimension
#define TILE_C 16

at::Tensor forward(
    at::Tensor x,
    int64_t dim,
    at::Tensor conv_weight,
    at::Tensor conv_bias
) {
    // 1) 3D convolution with unrolled kernel
    auto y = at::conv3d(x, conv_weight, conv_bias);
    // 2) Min along the specified dimension
    y = std::get<0>(y.min(dim));
    // 3) Softmax along the channel dimension (dim=1)
    y = at::softmax(y, 1);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Convolution + Min + Softmax (CPU)");
}