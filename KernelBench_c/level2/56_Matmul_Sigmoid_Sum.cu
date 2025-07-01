#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel to compute the linear transformation and accumulate the sum
template <typename scalar_t>
__global__ void my_kernel(
    const scalar_t* __restrict__ x,           // [batch_size, input_size]
    const scalar_t* __restrict__ weight,      // [hidden_size, input_size]
    const scalar_t* __restrict__ bias,        // [hidden_size]
    scalar_t* __restrict__ output,            // [batch_size]
    int input_size,
    int hidden_size) {

    // Compute batch index
    int batch_idx = blockIdx.x;

    // Compute hidden index
    int hidden_idx = threadIdx.x;

    // Allocate shared memory for partial sums
    extern __shared__ double shared_sum[];

    // Initialize sum to zero
    double sum = 0.0;

    // Ensure hidden_idx is within bounds
    if (hidden_idx < hidden_size) {
        // Compute linear transformation
        double linear = static_cast<double>(bias[hidden_idx]);
        for (int i = 0; i < input_size; ++i) {
            linear += static_cast<double>(x[batch_idx * input_size + i]) *
                      static_cast<double>(weight[hidden_idx * input_size + i]);
        }
        // Apply sigmoid function
        double s = 1.0 / (1.0 + exp(-linear));
        sum = s;
    }

    // Store sum in shared memory
    shared_sum[hidden_idx] = sum;
    __syncthreads();

    // Reduction to compute the total sum for the batch
    if (hidden_idx == 0) {
        double batch_sum = 0.0;
        for (int i = 0; i < hidden_size; ++i) {
            batch_sum += shared_sum[i];
        }
        output[batch_idx] = static_cast<scalar_t>(batch_sum);
    }
}

// Host function to launch the CUDA kernel
torch::Tensor my_kernel_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // x: [batch_size, input_size]
    // weight: [hidden_size, input_size]
    // bias: [hidden_size]
    // Output: [batch_size, 1]

    // Get sizes
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size}, options);

    // Configure kernel launch parameters
    int threads = hidden_size;
    int blocks = batch_size;
    size_t shared_mem_size = threads * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "my_kernel_forward", ([&] {
        my_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int>(input_size),
            static_cast<int>(hidden_size));

        // Check for errors during kernel launch
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    }));

    // Reshape output to [batch_size, 1]
    return output.view({batch_size, 1});
}

// PyBind wrapper
torch::Tensor forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // Check for CUDA tensors
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    // Ensure tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Call the CUDA kernel
    return my_kernel_forward(x, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Module function forward (CUDA)");
}