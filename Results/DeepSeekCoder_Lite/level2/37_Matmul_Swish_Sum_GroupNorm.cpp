#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// Fused kernel: applies swish activation + bias addition and then group normalization
// x     : input tensor (in/out) with shape (N, C)
// bias  : bias to add after swish activation (length C)
// gamma : group norm scale parameter (length C)
// beta  : group norm shift parameter (length C)
// N     : batch size
// C     : number of channels (out_features from linear)
// num_groups : number of groups for group normalization
// epsilon: small constant for numerical stability

void fused_swish_bias_groupnorm_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N,
    int C,
    int num_groups,
    float epsilon,
    int sample_idx,
    int group_idx,
    int group_channels,
    int block_size,
    int num_threads) {

    // base index for the current sample's channel and group offset
    // x is laid out as [N x C] so row-major
    int base = sample_idx * C + group_idx * group_channels;

    // Allocate external shared memory:
    // - First portion: group_channels floats for storing intermediate swish+bias results
    // - Next: block_size floats for partial sums
    // - Next: block_size floats for partial sum-of-squares
    std::vector<float> shared_mem(block_size + 2 * block_size);
    float* smem = shared_mem.data();                   // size: group_channels floats
    float* ssum = shared_mem.data() + group_channels;  // size: block_size floats
    float* ssumsq = ssum + block_size;                 // size: block_size floats

    // Additionally, use a couple of __shared__ variables for the final mean and inv_std
    __shared__ float group_mean;
    __shared__ float group_inv_std;

    // Step 1: Each thread loads its assigned elements from x, applies swish activation and bias addition,
    // and stores the result into shared memory.
    for (int i = threadIdx.x; i < group_channels; i += block_size) {
        int idx = base + i;
        // Load the input element
        float val = x[idx];
        // Compute swish activation: val * sigmoid(val). Use fast __expf.
        float sigmoid_val = 1.0f / (1.0f + exp(-val));
        float activated = val * sigmoid_val;
        
        // Compute the channel index for bias: since x has shape (N, C), the effective channel index
        // is given by (group_idx * group_channels + i).
        int channel = group_idx * group_channels + i;
        // Add bias using read-only __ldg
        activated += bias[channel];

        // Store the computed value in shared memory
        smem[i] = activated;
    }

    __sync_threads();

    // Step 2: Compute block-level reduction using warp-level primitives
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    // Each thread sums over its subset of the group's elements stored in smem
    for (int i = threadIdx.x; i < group_channels; i += block_size) {
        float v = smem[i];
        local_sum += v;
        local_sumsq += v * v;
    }

    // Perform warp-level reduction using shuffle operations
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
        local_sumsq += __shfl_down_sync(mask, local_sumsq, offset);
    }

    // Write warp-level reduced results to shared memory
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        ssum[warp_id] = local_sum;
        ssumsq[warp_id] = local_sumsq;
    }
    __sync_threads();

    // Final reduction across warp leaders by thread 0
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        float sumsq = 0.0f;
        int num_warps = (block_size + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            sum += ssum[i];
            sumsq += ssumsq[i];
        }
        float mean = sum / group_channels;
        float var = sumsq / group_channels - mean * mean;
        group_mean = mean;
        group_inv_std = 1.0f / sqrt(var + epsilon);
    }
    __sync_threads();

    // Step 3: Normalize the swish+bias outputs stored in shared memory and apply per-channel scale and shift
    for (int i = threadIdx.x; i < group_channels; i += block_size) {
        // Normalize: (value - mean) * inv_std
        float norm = (smem[i] - group_mean) * group_inv_std;
        // Determine channel index (same as used for bias) for group norm parameters
        int channel = group_idx * group_channels + i;
        // Apply gamma (scale) and beta (shift) using read-only __ldg
        norm = norm * gamma[channel] + beta[channel];
        
        // Write back the normalized value to global memory
        x[base + i] = norm;
    }
}

// Forward function: performs a linear operation followed by the fused activation and normalization.
// It takes in input tensors and parameters and returns the updated tensor.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor weight_bias,
    torch::Tensor bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int num_groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(weight_bias);
    CHECK_INPUT(bias);
    CHECK_INPUT(group_norm_weight);
    CHECK_INPUT(group_norm_bias);

    // Perform matrix multiplication (linear layer)
    auto x_linear = torch::linear(x, weight, weight_bias);

    // Get dimensions from x_linear. Expected shape: (N, C)
    int N = x_linear.size(0);
    int C = x_linear.size(1);

    // Configure kernel launch: one block per (sample, group)
    int blocks = N * num_groups;
    int threads = 128;  // Tunable block size
    
    // Compute group_channels. Assumes C is divisible by num_groups.
    int group_channels = C / num_groups;
    
    // Determine required shared memory size:
    // shared memory = group_channels (for swish+bias results) + 2 * threads (for reduction)
    size_t shared_mem_size = group_channels * sizeof(float) + 2 * threads * sizeof(float);

    // Launch the fused kernel
    #pragma omp parallel for
    for (int sample_idx = 0; sample_idx < N; ++sample_idx) {
        for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
            fused_swish_bias_groupnorm_kernel(
                x_linear.data_ptr<float>(),
                bias.data_ptr<float>(),
                group_norm_weight.data_ptr<float>(),
                group_norm_bias.data_ptr<float>(),
                N, C, num_groups, 1e-5f,
                sample_idx, group_idx, group_channels, threads, omp_get_num_threads());
        }
    }

    return x_linear;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Swish activation, bias addition, and group normalization");
}