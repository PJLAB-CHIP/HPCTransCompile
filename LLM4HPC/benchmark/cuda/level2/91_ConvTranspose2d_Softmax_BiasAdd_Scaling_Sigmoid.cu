#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Choose a BLOCK_SIZE that is optimal for the target hardware
#define BLOCK_SIZE 128

// This kernel minimizes warp divergence by avoiding divergent branching within warps.
template <typename scalar_t>
__global__ void optimized_fused_ops_kernel_minimized_warp_divergence(
    scalar_t* output,
    const scalar_t* conv_output,
    const scalar_t* channel_bias,
    float scaling_factor,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    const int spatial_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int spatial_size = height * width;
    if (spatial_pos >= batch_size * spatial_size) return;

    const int b = spatial_pos / spatial_size;
    const int pos = spatial_pos % spatial_size;
    const int h = pos / width;
    const int w = pos % width;

    // Shared memory to improve memory access
    __shared__ scalar_t shared_exp[BLOCK_SIZE];
    __shared__ scalar_t shared_max[BLOCK_SIZE];
    
    // Initialize local maximum
    scalar_t thread_max = -INFINITY;
    scalar_t thread_sum = 0;
    
    // Pre-compute base index to avoid redundant calculations
    const int base_idx = (b * channels * height + h) * width + w;
    
    // First pass: Find maximum while prefetching data
    #pragma unroll 4
    for (int c = 0; c < channels; ++c) {
        const int idx = base_idx + c * height * width;
        scalar_t val = conv_output[idx];
        thread_max = max(thread_max, val);
    }
    
    // Reduce maximum within warp and block
    shared_max[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Parallel reduction for maximum
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = max(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    const scalar_t max_val = shared_max[0];
    
    // Second pass: Compute exponentials and sum
    #pragma unroll 4
    for (int c = 0; c < channels; ++c) {
        const int idx = base_idx + c * height * width;
        scalar_t val = exp(conv_output[idx] - max_val);
        shared_exp[threadIdx.x] = val;
        thread_sum += val;
    }
    
    __syncthreads();
    
    // Final pass: Apply softmax, bias, scaling, and sigmoid
    #pragma unroll 4
    for (int c = 0; c < channels; ++c) {
        const int idx = base_idx + c * height * width;
        scalar_t val = shared_exp[threadIdx.x] / thread_sum;
        val = val + channel_bias[c];
        val *= scaling_factor;
        output[idx] = 1.0f / (1.0f + exp(-val));
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int stride,
    int padding,
    int output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias,
    float scaling_factor) {
    
    // Perform transposed convolution using PyTorch
    auto conv_out = torch::nn::functional::conv_transpose2d(
        x, conv_transpose,
        torch::nn::functional::ConvTranspose2dFuncOptions()
            .bias(conv_transpose_bias)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
    );

    TORCH_CHECK(bias.size(0) == conv_out.size(1), "Bias size must match channel dimension");
    
    auto output = torch::empty_like(conv_out);
    const int batch_size = conv_out.size(0);
    const int channels = conv_out.size(1);
    const int height = conv_out.size(2);
    const int width = conv_out.size(3);

    const int total_spatial = batch_size * height * width;
    const int blocks = (total_spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(conv_out.scalar_type(), "optimized_fused_ops_kernel_minimized_warp_divergence", ([&] {
        optimized_fused_ops_kernel_minimized_warp_divergence<scalar_t><<<blocks, BLOCK_SIZE>>>(
            output.data_ptr<scalar_t>(),
            conv_out.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            scaling_factor,
            batch_size,
            channels,
            height,
            width
        );
    }));

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Fused ConvTranspose2d+Softmax+Bias+Scale+Sigmoid with minimized warp divergence");
}
