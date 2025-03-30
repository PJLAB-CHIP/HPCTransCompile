#include <torch/extension.h>
#include <omp.h>
#include <math.h>
#include <float.h>
#include <vector>

// GELU activation function (approximation used in PyTorch)
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + coef * x * x)));
    return x * cdf;
}

// Fused kernel: Performs linear transformation, applies GELU activation, and softmax normalization
// Leverages shared memory to store the input row, which is reused for all dot-product computations,
// and uses shared memory for softmax reduction to minimize global memory latency.
__global__ void fused_shared_mem_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    // We expect the number of threads per block to be a padded value (multiple of 32) >= out_features
    int padded = blockDim.x;  

    // Shared memory layout: first region for storing the input row, second for softmax reduction
    // Total shared memory allocated: (in_features + padded) * sizeof(float)
    extern __shared__ float shared_mem[];
    float* s_x = shared_mem;             // Size: in_features (to store one row of input x)
    float* s_softmax = shared_mem + in_features; // Size: padded (for softmax reduction)

    int row = blockIdx.x;   // Each block processes one row of the batch
    int tid = threadIdx.x;

    // 1. Load the input row from global memory into shared memory
    for (int i = tid; i < in_features; i += padded) {
        s_x[i] = x[row * in_features + i];
    }
    __syncthreads();

    // 2. Compute the dot product for the linear transformation for each valid output feature
    float act = 0.0f;
    if (tid < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; k++) {
            sum += s_x[k] * weight[tid * in_features + k];
        }
        sum += bias[tid];
        act = gelu(sum);
        s_softmax[tid] = act;  // Store the activated value for softmax reduction
    } else {
        // For padded threads, use a sentinel value for max reduction
        s_softmax[tid] = -FLT_MAX;
    }
    __syncthreads();

    // 3. Reduction to compute the maximum activated value across the outputs (for softmax numerical stability)
    for (int stride = padded / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float other = s_softmax[tid + stride];
            s_softmax[tid] = (other > s_softmax[tid]) ? other : s_softmax[tid];
        }
        __syncthreads();
    }
    float row_max = s_softmax[0];
    __syncthreads();
    
    // 4. Compute the exponentials; invalid threads (tid >= out_features) produce 0
    float exp_val = 0.0f;
    if (tid < out_features) {
        exp_val = expf(act - row_max);
        s_softmax[tid] = exp_val;
    } else {
        s_softmax[tid] = 0.0f;
    }
    __syncthreads();

    // 5. Reduction to compute the sum of exponentials
    for (int stride = padded / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_softmax[tid] += s_softmax[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = s_softmax[0];
    __syncthreads();

    // 6. Write the normalized softmax result for valid output features
    if (tid < out_features) {
        output[row * out_features + tid] = exp_val / sum_exp;
    }
}

// Forward function that wraps the kernel launch
// It sets up the padded thread count and allocates shared memory for both the input row and softmax reduction buffer
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, out_features}, options);

    // Determine padded thread count: round up out_features to the next multiple of 32
    int threads = ((out_features + 31) / 32) * 32;
    dim3 blocks(batch_size);
    dim3 threadBlock(threads);

    // Shared memory size: space for one input row (in_features floats) + softmax buffer (threads floats)
    int shared_mem_size = (in_features + threads) * sizeof(float);

    fused_shared_mem_kernel<<<blocks, threadBlock, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Linear + GELU + Softmax forward with shared memory");
}