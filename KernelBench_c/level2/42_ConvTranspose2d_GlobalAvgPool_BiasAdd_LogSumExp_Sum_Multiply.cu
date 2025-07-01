#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

__global__ void sync_optimized_kernel(
    const float* input,
    const float* conv_weights,
    const float* conv_bias,
    const float* bias,
    float* output,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    __shared__ float shared_data[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * BLOCK_SIZE + tid;
    
    // Load input and perform initial computations with minimal synchronizations
    float val = 0.0f;
    if (idx < batch_size * channels * height * width) {
        val = input[idx];
        for (int i = 0; i < channels; ++i) {
            val += conv_weights[i] * val;
        }
        val += conv_bias[idx % channels];
        
        // Utilize shared memory for intermediate results without unnecessary synchronizations
        atomicAdd(&shared_data[tid % channels], val / (height * width));
    }
    __syncthreads();
    
    // Only synchronize here, post accumulation.
    if (tid < channels) {
        shared_data[tid] += bias[tid];
        shared_data[tid] = expf(shared_data[tid]);
    }
    __syncthreads();

    // Final reduction and output computation with minimal synchronization
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < channels; ++i) {
            sum += shared_data[i];
        }
        output[bid] = logf(sum) * 10.0f;
    }
}

torch::Tensor module_fn(
    torch::Tensor x,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(conv_transpose.is_cuda(), "conv_transpose must be a CUDA tensor");
    TORCH_CHECK(conv_transpose_bias.is_cuda(), "conv_transpose_bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    
    auto output = torch::empty({batch_size, 1}, x.options());
    
    dim3 blocks((batch_size * channels * height * width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);
    
    sync_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        conv_transpose.data_ptr<float>(),
        conv_transpose_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Sync optimized CUDA kernel forward");
}
