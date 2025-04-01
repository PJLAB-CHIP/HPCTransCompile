#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
namespace py = pybind11;
namespace {

template <typename scalar_t>
__global__ void apply_min_clamp_kernel(
    scalar_t* output,
    const scalar_t* input,
    const scalar_t min_value,
    const scalar_t max_value,
    int64_t num_elements) {
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        scalar_t val = input[idx];
        val = min(val, min_value);
        val = max(val, min_value);
        val = min(val, max_value);
        output[idx] = val;
    }
}

} // namespace

torch::Tensor forward(
    torch::Tensor x,
    py::object params_obj,
    int64_t groups,
    float min_value,
    float max_value,
    float dropout_p,
    bool training) {
    
    torch::Tensor conv_weight = params_obj["conv_weight"].cast<torch::Tensor>();
    torch::Tensor conv_bias = params_obj["conv_bias"].cast<torch::Tensor>();
    torch::Tensor norm_weight = params_obj["norm_weight"].cast<torch::Tensor>();
    torch::Tensor norm_bias = params_obj["norm_bias"].cast<torch::Tensor>();
    // Conv3d
    auto x_conv = torch::conv3d(x, conv_weight, conv_bias);

    // GroupNorm
    auto x_norm = torch::group_norm(x_conv, groups, norm_weight, norm_bias);

    // Apply min and clamp
    auto output = x_norm.clone();
    int64_t num_elements = output.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "min_clamp_kernel", ([&] {
        apply_min_clamp_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            x_norm.data_ptr<scalar_t>(),
            static_cast<scalar_t>(min_value),
            static_cast<scalar_t>(max_value),
            num_elements);
    }));

    // Apply dropout if training
    if (training && dropout_p > 0.0f) {
        // x = F.dropout(x, p=dropout_p)
        auto output_dropout = torch::nn::functional::dropout(output, torch::nn::functional::DropoutFuncOptions().p(dropout_p).inplace(false));
        output = output_dropout;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CUDA forward");
}
