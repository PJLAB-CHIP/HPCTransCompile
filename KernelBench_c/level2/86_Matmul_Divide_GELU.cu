#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

// Device function for GELU calculation
__device__ __forceinline__ float gelu_impl(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
    return x * cdf;
}

__global__ void block_optimized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight_t,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int input_size,
    const int output_size,
    const float divisor
) {
    __shared__ float shared_x[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_weight[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;

    float acc = 0.0f;
    if (row < batch_size && col < output_size) {
        acc = bias[col];
    }

    const int num_tiles = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    #pragma unroll 4
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int input_row = row;
        const int input_col = tile * BLOCK_SIZE + tx;
        const int weight_row = tile * BLOCK_SIZE + ty;
        const int weight_col = col;

        // Load input tile
        if (input_row < batch_size && input_col < input_size) {
            shared_x[ty][tx] = x[input_row * input_size + input_col];
        } else {
            shared_x[ty][tx] = 0.0f;
        }

        // Load weight tile
        if (weight_row < input_size && weight_col < output_size) {
            shared_weight[ty][tx] = weight_t[weight_row * output_size + weight_col];
        } else {
            shared_weight[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += shared_x[ty][k] * shared_weight[k][tx];
        }

        __syncthreads();
    }

    if (row < batch_size && col < output_size) {
        acc /= divisor;
        output[row * output_size + col] = gelu_impl(acc);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    x = x.contiguous().cuda();
    weight = weight.contiguous().cuda();
    bias = bias.contiguous().cuda();
    
    auto weight_t = weight.transpose(0, 1).contiguous();
    
    const int batch_size = x.size(0);
    const int input_size = x.size(1);
    const int output_size = weight.size(0);
    
    auto output = torch::empty({batch_size, output_size}, x.options());
    
    // Use 16x16 thread blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    block_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Block size optimized fused kernel");
}