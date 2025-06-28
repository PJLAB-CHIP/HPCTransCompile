#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for combined linear, subtract, multiply and ReLU operations
template <typename scalar_t>
__global__ void linear_subtract_multiply_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float subtract_value,
    const float multiply_value) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        scalar_t sum = 0;
        
        // Compute linear transformation
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        
        // Add bias
        sum += bias[col];
        
        // Subtract and multiply
        sum = (sum - subtract_value) * multiply_value;
        
        // ReLU activation
        sum = sum > 0 ? sum : 0;
        
        output[row * out_features + col] = sum;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (batch_size + threads.x - 1) / threads.x,
        (out_features + threads.y - 1) / threads.y
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "linear_subtract_multiply_relu_kernel", ([&] {
        linear_subtract_multiply_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features, 
            out_features,
            subtract_value,
            multiply_value
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Linear transform with subtract, multiply and ReLU forward");
}