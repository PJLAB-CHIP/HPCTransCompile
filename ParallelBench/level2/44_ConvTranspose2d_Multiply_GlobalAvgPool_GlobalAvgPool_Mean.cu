#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void optimized_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const float multiplier
) {
    extern __shared__ float sdata[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = bid / C;
    const int channel_idx = bid % C;
    const int spatial_size = H * W;
    
    // Calculate input offset for this block
    const float* block_input = input + (batch_idx * C * spatial_size) + (channel_idx * spatial_size);
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Process multiple elements per thread with stride pattern and apply multiplier
    #pragma unroll 8  // Increased unroll factor
    for (int i = tid; i < spatial_size; i += BLOCK_SIZE) {
        sum += block_input[i] * multiplier;
    }
    
    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();
    
    // Two-phase reduction for better performance
    if (BLOCK_SIZE >= 1024) {
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within a warp)
    if (tid < 32) {
        volatile float* vmem = sdata;
        if (BLOCK_SIZE >= 64) vmem[tid] += vmem[tid + 32];
        if (BLOCK_SIZE >= 32) vmem[tid] += vmem[tid + 16];
        if (BLOCK_SIZE >= 16) vmem[tid] += vmem[tid + 8];
        if (BLOCK_SIZE >= 8)  vmem[tid] += vmem[tid + 4];
        if (BLOCK_SIZE >= 4)  vmem[tid] += vmem[tid + 2];
        if (BLOCK_SIZE >= 2)  vmem[tid] += vmem[tid + 1];
    }
    
    // First thread writes result
    if (tid == 0) {
        output[bid] = sdata[0] / (spatial_size);  // Normalize during reduction
    }
}

at::Tensor module_fn(
    at::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    at::Tensor conv_transpose,
    at::Tensor conv_transpose_bias,
    double multiplier
) {
    // Apply transposed convolution
    at::Tensor y = at::conv_transpose2d(
        x,
        conv_transpose,
        conv_transpose_bias,
        {stride, stride},
        {padding, padding},
        {output_padding, output_padding},
        1,
        {1, 1}
    );
    
    // Prepare output tensor
    auto options = torch::TensorOptions().device(y.device()).dtype(y.dtype());
    auto dims = y.sizes();
    at::Tensor output = torch::zeros({dims[0], dims[1]}, options);
    
    // Launch kernel with optimized configuration
    constexpr int BLOCK_SIZE = 512;  // Optimized block size
    const int blocks = dims[0] * dims[1];
    const int shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    optimized_reduction_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, shared_mem_size>>>(
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        dims[0], dims[1], dims[2], dims[3],
        static_cast<float>(multiplier)
    );
    
    // Compute final mean
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Module function");
}