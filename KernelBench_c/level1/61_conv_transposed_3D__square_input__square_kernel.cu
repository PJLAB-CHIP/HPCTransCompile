#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    // Perform the transposed 3D convolution
    auto output = torch::conv_transpose3d(
        input, 
        weight, 
        bias, 
        {stride, stride, stride},
        {padding, padding, padding},
        {output_padding, output_padding, output_padding},
        groups,
        {1, 1, 1}  // dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Transposed 3D convolution forward (CUDA)");
}