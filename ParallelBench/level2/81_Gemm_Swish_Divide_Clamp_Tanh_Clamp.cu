#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macros for input checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// CUDA kernel using 2D grid and block indexing for natural mapping to 2D data
template <typename scalar_t>
__global__ void module_kernel_2d(
    const scalar_t* __restrict__ x_in,
    scalar_t* __restrict__ x_out,
    int height,
    int width) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int index = row * width + col;
        scalar_t x = x_in[index];

        // Swish activation: x = x * sigmoid(x)
        scalar_t sigmoid_x = static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + exp(-x));
        x = x * sigmoid_x;

        // Divide by 2
        x = x / static_cast<scalar_t>(2);

        // Clamp between -1 and 1
        x = max(min(x, static_cast<scalar_t>(1)), static_cast<scalar_t>(-1));

        // Tanh activation
        x = tanh(x);

        // Clamp again between -1 and 1
        x = max(min(x, static_cast<scalar_t>(1)), static_cast<scalar_t>(-1));

        x_out[index] = x;
    }
}

// CUDA forward function
torch::Tensor module_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Execute linear operation: x_linear = F.linear(x, weight, bias)
    auto x_linear = torch::addmm(bias, x, weight.t());
    auto x_out = torch::empty_like(x_linear);

    // Assuming x_linear is a 2D matrix
    int height = x_linear.size(0);
    int width = x_linear.size(1);

    // Define 2D block dimensions. 16x16 is a common choice for 2D mapping
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_linear.scalar_type(), "module_forward_cuda", ([&] {
        module_kernel_2d<scalar_t><<<grid, block>>>(
            x_linear.data_ptr<scalar_t>(),
            x_out.data_ptr<scalar_t>(),
            height,
            width);
    }));

    return x_out;
}

// C++ interface
torch::Tensor module_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    return module_forward_cuda(x, weight, bias);
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_forward, "Custom module forward function (CUDA)");
}
