// fused_scale_bn_coop.cu
// This CUDA code fuses scaling and batch normalization in a single kernel launch per feature.
// Each block processes one feature column of the input (with dimensions [batch_size, features]).
// It first computes the mean and variance (after applying scaling) using warp-level reductions,
// updates the running statistics, and then performs the elementwise normalization in a second pass.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Fused kernel: each block processes one feature (channel).
// input: pointer to linear output of shape (batch_size, features)
// scale: scaling factors (per feature)
// running_mean, running_var: running statistics to be updated (per feature)
// gamma, beta: BatchNorm weight and bias (per feature)
// output: result after fused scaling and batchnorm, same shape as input
// eps: epsilon for numerical stability in batch norm
// momentum: momentum for running stats update
__global__ void fused_scale_bn_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int features,
    float eps,
    float momentum) {

    // Each block handles one feature (channel)
    int c = blockIdx.x;
    if (c >= features) return;

    // Load per-feature parameters into registers
    float s = scale[c];   // scaling factor
    float g = gamma[c];   // BN weight
    float b = beta[c];    // BN bias

    // First pass: compute scaled sum and sum of squares over the batch dimension
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Process batch elements in a grid-stride loop over the batch dimension
    for (int n = threadIdx.x; n < batch_size; n += blockDim.x) {
        // Each element is scaled before statistics are computed
        float val = input[n * features + c] * s;
        sum += val;
        sum_sq += val * val;
    }

    // Use warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
        sum_sq += __shfl_down_sync(mask, sum_sq, offset);
    }

    // Declare shared memory (assuming at most 32 warps per block)
    __shared__ float shared_sum[32];
    __shared__ float shared_sum_sq[32];

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared_sum[warp_id] = sum;
        shared_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp aggregates results from all warps
    float total_sum = 0.0f;
    float total_sum_sq = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (threadIdx.x < num_warps) {
        total_sum = shared_sum[threadIdx.x];
        total_sum_sq = shared_sum_sq[threadIdx.x];
    }

    if (threadIdx.x < WARP_SIZE) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            total_sum += __shfl_down_sync(mask, total_sum, offset);
            total_sum_sq += __shfl_down_sync(mask, total_sum_sq, offset);
        }
        if (threadIdx.x == 0) {
            // Final computed mean and variance for feature c
            float mean = total_sum / batch_size;
            float var = total_sum_sq / batch_size - mean * mean;
            
            // Update running statistics (in-place update since one block per feature)
            running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
            running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;
            // Reuse shared memory slot to store computed mean and var for normalization
            shared_sum[0] = mean;
            shared_sum_sq[0] = var;
        }
    }
    __syncthreads();

    // Retrieve computed mean and variance from shared memory
    float mean = shared_sum[0];
    float var = shared_sum_sq[0];
    float inv_std = rsqrtf(var + eps);

    // Second pass: perform fused normalization with scaling
    // Formula: output = ((input * s - mean) * inv_std) * g + b
    for (int n = threadIdx.x; n < batch_size; n += blockDim.x) {
        float val = input[n * features + c] * s;
        output[n * features + c] = ((val - mean) * inv_std) * g + b;
    }
}

// Host forward function
at::Tensor forward(
    at::Tensor x,
    float eps,
    float momentum,
    at::Tensor running_mean,
    at::Tensor running_var,
    at::Tensor gemm_weight,
    at::Tensor gemm_bias,
    at::Tensor scale,
    at::Tensor gamma,   // BatchNorm weight
    at::Tensor beta     // BatchNorm bias
) {
    auto device = x.device();
    
    // Ensure tensors are contiguous and on the proper device
    x = x.contiguous();
    gemm_weight = gemm_weight.to(device).contiguous();
    gemm_bias = gemm_bias.to(device).contiguous();
    scale = scale.to(device).contiguous();
    gamma = gamma.to(device).contiguous();
    beta = beta.to(device).contiguous();
    running_mean = running_mean.to(device).contiguous();
    running_var = running_var.to(device).contiguous();

    // Compute linear output (matrix multiply + bias)
    auto linear_output = at::linear(x, gemm_weight, gemm_bias);
    auto output = at::empty_like(linear_output);

    int batch_size = linear_output.size(0);
    int features = linear_output.size(1);

    // Launch one block per feature
    int threads = 256;
    dim3 blocks(features);

    fused_scale_bn_kernel<<<blocks, threads>>>(
        linear_output.data_ptr<float>(),
        scale.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        eps,
        momentum
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused scale and BN forward (CUDA)");
}
