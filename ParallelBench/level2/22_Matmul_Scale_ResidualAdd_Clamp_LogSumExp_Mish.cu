#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Define block size for kernels
constexpr int BLOCK_SIZE = 256;

// Scalar kernel using __ldg() for read-only loads
__global__ void clamp_and_scale_scalar(const float* __restrict__ in, float* __restrict__ out, int num_elements, float factor, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Use __ldg() to load from global memory in read-only cache
        float v = __ldg(&in[idx]);
        v = v * (2.0f * factor);
        v = fminf(fmaxf(v, min_val), max_val);
        out[idx] = v;
    }
}

// Vectorized kernel processing 4 floats at a time using float4
__global__ void clamp_and_scale_vectorized(const float4* __restrict__ in, float4* __restrict__ out, int num_elements4, float factor, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements4) {
        // Load a vector of 4 floats using __ldg()
        float4 v = __ldg(&in[idx]);
        float s = 2.0f * factor;
        v.x = fminf(fmaxf(v.x * s, min_val), max_val);
        v.y = fminf(fmaxf(v.y * s, min_val), max_val);
        v.z = fminf(fmaxf(v.z * s, min_val), max_val);
        v.w = fminf(fmaxf(v.w * s, min_val), max_val);
        out[idx] = v;
    }
}

// Kernel to perform LogSumExp across rows and apply Mish activation
__global__ void logsumexp_mish_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x; // each block works on one row
    int tid = threadIdx.x;

    // Find maximum value in the row using __ldg() to load read-only values
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = __ldg(&input[row * cols + i]);
        max_val = fmaxf(max_val, val);
    }
    sdata[tid] = max_val;
    __syncthreads();
    // Reduction over shared memory to obtain the row maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // Compute the sum of exp(value - max) for numerical stability
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = __ldg(&input[row * cols + i]);
        sum += expf(v - row_max);
    }
    sdata[tid] = sum;
    __syncthreads();
    // Reduction to sum up all the values
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float lse = logf(sdata[0]) + row_max;
    
    // Apply Mish activation: mish(x) = x * tanh(softplus(x)) => final: x * (x * tanh(softplus(x)))
    float softplus = log1pf(expf(lse));
    float mish = lse * tanhf(softplus);
    output[row] = lse * mish;
}

// Forward function that implements the complete fused operation
torch::Tensor module_fn_forward(
    torch::Tensor x,
    float scale_factor,
    float clamp_min,
    float clamp_max,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Ensure inputs are contiguous for aligned memory accesses
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // 1. Matrix multiplication and bias addition
    auto out = torch::mm(x, weight.transpose(0, 1));
    out.add_(bias);

    // 2. Fuse scaling, residual addition, and clamping using a custom kernel
    int num_elements = out.numel();
    // Check for 128-bit alignment and divisibility by 4 for vectorized operations
    bool use_vectorized = (num_elements % 4 == 0) && (((uintptr_t)out.data_ptr<float>()) % 16 == 0);

    if (use_vectorized) {
        int num_elements4 = num_elements / 4;
        int blocks = (num_elements4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        clamp_and_scale_vectorized<<<blocks, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(out.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            num_elements4,
            scale_factor,
            clamp_min,
            clamp_max);
    } else {
        int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        clamp_and_scale_scalar<<<blocks, BLOCK_SIZE>>>(
            out.data_ptr<float>(),
            out.data_ptr<float>(),
            num_elements,
            scale_factor,
            clamp_min,
            clamp_max);
    }
    
    // 3. Apply LogSumExp and Mish activation along rows using a reduction kernel
    auto output = torch::empty({out.size(0), 1}, out.options());
    int shared_mem = BLOCK_SIZE * sizeof(float);
    logsumexp_mish_kernel<<<out.size(0), BLOCK_SIZE, shared_mem>>>(
        out.data_ptr<float>(),
        output.data_ptr<float>(),
        out.size(0),
        out.size(1));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Forward pass for module_fn (CUDA)");
}
