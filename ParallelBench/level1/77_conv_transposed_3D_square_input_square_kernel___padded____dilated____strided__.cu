```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

    // Perform the 3D transposed convolution using PyTorch's built-in function
    auto output = torch::conv_transpose3d(
        input, 
        weight, 
        bias, 
        stride, 
        padding, 
        /*output_padding=*/0, 
        /*groups=*/1, 
        dilation
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "3D transposed convolution forward (CUDA)");
}