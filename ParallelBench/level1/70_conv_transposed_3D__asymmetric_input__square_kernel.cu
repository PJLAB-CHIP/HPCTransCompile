```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    int64_t dilation) {

    return torch::conv_transpose3d(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D transposed convolution forward");
}
```