#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4
#define SHARED_MEM_SIZE (BLOCK_SIZE * ELEMENTS_PER_THREAD)

__device__ __forceinline__ float fused_mish_tanh_activation(float x) {
    float softplus = logf(1.0f + expf(x));
    float mish = x * tanhf(softplus);
    return tanhf(mish);
}

__global__ void shared_mem_mish_tanh_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int total_elements
) {
    __shared__ float shared_data[SHARED_MEM_SIZE];
    
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * SHARED_MEM_SIZE + tid;
    
    // Load data into shared memory in coalesced manner
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = global_idx + i * BLOCK_SIZE;
        if (idx < total_elements) {
            shared_data[tid + i * BLOCK_SIZE] = input[idx];
        }
    }
    
    // Only synchronize after shared memory writes
    __syncthreads();
    
    // Process data in shared memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = global_idx + i * BLOCK_SIZE;
        if (idx < total_elements) {
            const float val = shared_data[tid + i * BLOCK_SIZE];
            const float result = fused_mish_tanh_activation(val);
            output[idx] = result;
        }
    }
    
    // No need for synchronization here since we're done with shared memory
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "Convolution weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "Convolution bias must be a CUDA tensor");

    auto x_conv = at::conv3d(
        x,
        conv_weight,
        conv_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    auto output = torch::empty_like(x_conv);
    const int total_elements = x_conv.numel();
    
    // Calculate grid size based on total elements and shared memory usage
    const int num_blocks = (total_elements + SHARED_MEM_SIZE - 1) / SHARED_MEM_SIZE;
    
    shared_mem_mish_tanh_kernel<<<num_blocks, BLOCK_SIZE>>>(
        output.data_ptr<float>(),
        x_conv.data_ptr<float>(),
        total_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Shared memory optimized convolution with Mish and Tanh activations (CUDA)");
}