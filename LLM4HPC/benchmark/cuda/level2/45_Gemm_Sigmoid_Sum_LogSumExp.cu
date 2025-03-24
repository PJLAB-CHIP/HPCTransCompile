#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function for computing dot product
__device__ __forceinline__ float dot_product(
    const float* __restrict__ vec1,
    const float* __restrict__ vec2,
    const int size
) {
    float result = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < size; ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Device function for sigmoid activation
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for parallel reduction in shared memory
__device__ __forceinline__ float block_reduce_sum(float val, float* shared, const int tid) {
    shared[tid] = val;
    __syncthreads();

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    return shared[0];
}

// Device function for parallel max reduction
__device__ __forceinline__ float block_reduce_max(float val, float* shared, const int tid) {
    shared[tid] = val;
    __syncthreads();

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void fused_gemm_sigmoid_logsumexp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int input_size,
    const int hidden_size
) {
    extern __shared__ float shared_mem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= batch_size) return;

    float local_sum = 0.0f;
    const float* row_input = &input[row * input_size];

    for (int col = tid; col < hidden_size; col += blockDim.x) {
        const float* col_weight = &weight[col * input_size];
        float dot = dot_product(row_input, col_weight, input_size);
        dot += bias[col];
        local_sum += sigmoid(dot);
    }

    float row_total = block_reduce_sum(local_sum, shared_mem, tid);
    if (tid == 0) {
        output[row] = row_total;
    }

    // Synchronize between steps
    __syncthreads();
    float local_max = -INFINITY;
    for (int i = tid; i < batch_size; i += blockDim.x) {
        local_max = max(local_max, output[i]);
    }
    float max_val = block_reduce_max(local_max, shared_mem, tid);
    __syncthreads();

    // Compute sum of exp(x - max)
    float local_exp_sum = 0.0f;
    for (int i = tid; i < batch_size; i += blockDim.x) {
        local_exp_sum += expf(output[i] - max_val);
    }
    float sum_exp_val = block_reduce_sum(local_exp_sum, shared_mem, tid);

    if (tid == 0) {
        output[0] = logf(sum_exp_val) + max_val;
    }
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0);

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());

    auto final_output = torch::empty({1}, options);

    const int threads_per_block = 128;
    dim3 grid(batch_size);
    
    fused_gemm_sigmoid_logsumexp_kernel<<<grid, threads_per_block, threads_per_block * sizeof(float)>>> (
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        final_output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return final_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass");
}