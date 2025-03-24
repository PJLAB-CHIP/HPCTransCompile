#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// Optimized kernel using __ldg() for read-only data and aligned memory access
__global__ void rnn_forward_aligned_ldg_kernel(
    const float4* __restrict__ x4,      // [batch, input_size/4]
    const float4* __restrict__ h4,      // [batch, hidden_size/4]
    const float4* __restrict__ weight4, // [hidden_dim, (input_size+hidden_size)/4]
    const float* __restrict__ bias,    // [hidden_dim]
    float* __restrict__ output,        // [batch, hidden_dim]
    int input_size,
    int hidden_size
) {
    int batch = blockIdx.x;    
    int neuron = blockIdx.y;   
    int combined_dim = (input_size + hidden_size + 3) / 4; // Rounded up for float4

    // Shared memory for reduction
    extern __shared__ float shared_sum[];
    
    float local_sum = 0.0f;
    
    // Process input data with aligned float4 loads
    int input_blocks = (input_size + 3) / 4;
    for (int idx = threadIdx.x; idx < input_blocks; idx += blockDim.x) {
        float4 val = __ldg(&x4[batch * input_blocks + idx]);
        float4 w = __ldg(&weight4[neuron * combined_dim + idx]);
        
        // Handle partial float4 at boundary
        if (idx == input_blocks - 1 && (input_size % 4) != 0) {
            switch (input_size % 4) {
                case 1:
                    local_sum += val.x * w.x;
                    break;
                case 2:
                    local_sum += val.x * w.x + val.y * w.y;
                    break;
                case 3:
                    local_sum += val.x * w.x + val.y * w.y + val.z * w.z;
                    break;
            }
        } else {
            local_sum += val.x * w.x + val.y * w.y + val.z * w.z + val.w * w.w;
        }
    }

    // Process hidden state data with aligned float4 loads
    int hidden_blocks = (hidden_size + 3) / 4;
    int hidden_offset = input_blocks;
    for (int idx = threadIdx.x; idx < hidden_blocks; idx += blockDim.x) {
        float4 val = __ldg(&h4[batch * hidden_blocks + idx]);
        float4 w = __ldg(&weight4[neuron * combined_dim + hidden_offset + idx]);
        
        // Handle partial float4 at boundary
        if (idx == hidden_blocks - 1 && (hidden_size % 4) != 0) {
            switch (hidden_size % 4) {
                case 1:
                    local_sum += val.x * w.x;
                    break;
                case 2:
                    local_sum += val.x * w.x + val.y * w.y;
                    break;
                case 3:
                    local_sum += val.x * w.x + val.y * w.y + val.z * w.z;
                    break;
            }
        } else {
            local_sum += val.x * w.x + val.y * w.y + val.z * w.z + val.w * w.w;
        }
    }

    // Store in shared memory and synchronize
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within block using sequential addressing
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Final warp reduction
    if (threadIdx.x < 32) {
        volatile float* smem = shared_sum;
        if (blockDim.x > 64) smem[threadIdx.x] += smem[threadIdx.x + 32];
        if (blockDim.x > 32) smem[threadIdx.x] += smem[threadIdx.x + 16];
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        smem[threadIdx.x] += smem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        output[batch * hidden_size + neuron] = tanhf(shared_sum[0] + __ldg(&bias[neuron]));
    }
}

torch::Tensor module_fn(
    torch::Tensor x,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias,
    torch::Tensor hidden
) {
    x = x.contiguous();
    hidden = hidden.to(x.device()).contiguous();
    i2h_weight = i2h_weight.contiguous();
    i2h_bias = i2h_bias.contiguous();

    int batch = x.size(0);
    int input_size = x.size(1);
    int hidden_size = hidden.size(1);

    auto output = torch::empty({batch, hidden_size}, x.options());

    dim3 blocks(batch, hidden_size);
    int threads = 256;
    size_t shared_bytes = threads * sizeof(float);

    rnn_forward_aligned_ldg_kernel<<<blocks, threads, shared_bytes>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<const float4*>(hidden.data_ptr<float>()),
        reinterpret_cast<const float4*>(i2h_weight.data_ptr<float>()),
        i2h_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size,
        hidden_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "RNN forward with aligned loads and __ldg optimization (CUDA)");
}