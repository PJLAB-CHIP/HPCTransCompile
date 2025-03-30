#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>

template<int BLOCK_SIZE>
void optimized_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const float multiplier,
    int num_threads
) {
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int block_size = BLOCK_SIZE;
        int spatial_size = H * W;
        int shared_mem_size = block_size * sizeof(float);

        // Allocate shared memory
        __shared__ float sdata[512];  // Assuming BLOCK_SIZE >= 512 for simplicity

        // Initialize accumulator
        float sum = 0.0f;

        // Process multiple elements per thread with stride pattern and apply multiplier
        #pragma unroll 8  // Increased unroll factor
        for (int i = tid; i < spatial_size; i += block_size) {
            sum += input[(i / spatial_size) * C * spatial_size + (i % spatial_size)] * multiplier;
        }

        // Store in shared memory
        sdata[tid] = sum;
        __sync_thread_fence();  // Ensure all threads write before sync

        // Two-phase reduction for better performance
        if (block_size >= 1024) {
            if (tid < 512) sdata[tid] += sdata[tid + 512];
            __sync_thread_fence();
        }
        if (block_size >= 512) {
            if (tid < 256) sdata[tid] += sdata[tid + 256];
            __sync_thread_fence();
        }
        if (block_size >= 256) {
            if (tid < 128) sdata[tid] += sdata[tid + 128];
            __sync_thread_fence();
        }
        if (block_size >= 128) {
            if (tid < 64) sdata[tid] += sdata[tid + 64];
            __sync_thread_fence();
        }

        // Warp-level reduction (no sync needed within a warp)
        if (tid < 32) {
            volatile float* vmem = sdata;
            if (block_size >= 64) vmem[tid] += vmem[tid + 32];
            if (block_size >= 32) vmem[tid] += vmem[tid + 16];
            if (block_size >= 16) vmem[tid] += vmem[tid + 8];
            if (block_size >= 8)  vmem[tid] += vmem[tid + 4];
            if (block_size >= 4)  vmem[tid] += vmem[tid + 2];
            if (block_size >= 2)  vmem[tid] += vmem[tid + 1];
        }

        // First thread writes result
        if (tid == 0) {
            output[(blockIdx / C) * spatial_size + (blockIdx % C)] = sdata[0] / (spatial_size);  // Normalize during reduction
        }
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
    const int num_threads = omp_get_max_threads();

    optimized_reduction_kernel<BLOCK_SIZE>(
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        dims[0], dims[1], dims[2], dims[3],
        static_cast<float>(multiplier),
        num_threads
    );

    // Compute final mean
    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Module function");
}