#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel: leverages shared memory for frequently accessed data
__global__ void shared_memory_coalesced_access_kernel(
    const float* __restrict__ x,    // input tensor (result of conv2d), shape [N, C, H, W]
    float* __restrict__ y,          // output tensor, same shape
    const float* __restrict__ bias, // bias for elementwise op (either size 1 or C)
    const float* __restrict__ scale,// scale for elementwise op (either size 1 or C)
    const float* __restrict__ gn_weight, // group norm weight, shape [C]
    const float* __restrict__ gn_bias,   // group norm bias, shape [C]
    int N, int C, int H, int W,
    int num_groups,
    bool bias_broadcast,
    bool scale_broadcast,
    float eps) {

    int group_idx = blockIdx.x % num_groups;
    int sample_idx = blockIdx.x / num_groups;
    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;

    int sample_offset = sample_idx * C * H * W;
    int group_channel_offset = group_idx * channels_per_group;

    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sum_sq = shared_mem + blockDim.x;

    // Shared memory for storing bias and scale per group
    __shared__ float shared_bias[1024];
    __shared__ float shared_scale[1024];

    if (threadIdx.x < channels_per_group) {
        int c = group_channel_offset + threadIdx.x;
        shared_bias[threadIdx.x] = bias_broadcast ? bias[0] : bias[c];
        shared_scale[threadIdx.x] = scale_broadcast ? scale[0] : scale[c];
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int hw = i % (H * W);
        int c = group_channel_offset + c_local;
        int idx = sample_offset + c * (H * W) + hw;

        float in_val = __ldg(&x[idx]);
        float b_val = shared_bias[c_local];
        float s_val = shared_scale[c_local];
        float pre_act = (in_val + b_val) * s_val;
        float v = 1.0f / (1.0f + expf(-pre_act));  // sigmoid activation

        y[idx] = v;
        local_sum += v;
        local_sum_sq += v * v;
    }

    int tid = threadIdx.x;
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    float group_mean = 0.0f;
    float group_var = 0.0f;
    if (tid == 0) {
        group_mean = shared_sum[0] / group_size;
        group_var = shared_sum_sq[0] / group_size - group_mean * group_mean;
        shared_sum[0] = group_mean;
        shared_sum_sq[0] = group_var;
    }
    __syncthreads();

    group_mean = shared_sum[0];
    group_var = shared_sum_sq[0];
    float inv_std = 1.0f / sqrtf(group_var + eps);

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / (H * W);
        int hw = i % (H * W);
        int c = group_channel_offset + c_local;
        int idx = sample_offset + c * (H * W) + hw;

        float v = y[idx];
        float normalized = (v - group_mean) * inv_std;
        float gamma = __ldg(&gn_weight[c]);
        float beta = __ldg(&gn_bias[c]);
        y[idx] = gamma * normalized + beta;
    }
}

// Launcher for the shared memory optimized kernel
void shared_memory_coalesced_access_cuda(
    at::Tensor x,          // Input from conv2d
    at::Tensor bias,       // Bias for elementwise op
    at::Tensor scale,      // Scale for elementwise op
    at::Tensor y,          // Output tensor
    at::Tensor gn_weight,  // Group normalization weight
    at::Tensor gn_bias,    // Group normalization bias
    int64_t num_groups,
    bool bias_broadcast,
    bool scale_broadcast,
    float eps) {

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int channels_per_group = C / num_groups;
    int total_blocks = N * num_groups;
    int threads = 256;
    size_t shared_mem_size = (2 * threads + 2 * channels_per_group) * sizeof(float);

    shared_memory_coalesced_access_kernel<<<total_blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        N, C, H, W,
        num_groups,
        bias_broadcast,
        scale_broadcast,
        eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel failed : ", cudaGetErrorString(err));
    }
}

// Forward function
at::Tensor module_fn_forward(
    at::Tensor x,
    at::Tensor conv_weight,
    at::Tensor conv_bias,
    at::Tensor bias,
    at::Tensor scale,
    at::Tensor gn_weight,
    at::Tensor gn_bias,
    int64_t num_groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(conv_weight);
    if (conv_bias.defined()) CHECK_INPUT(conv_bias);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);
    CHECK_INPUT(gn_weight);
    CHECK_INPUT(gn_bias);

    x = at::conv2d(x, conv_weight, conv_bias);

    at::Tensor y = at::empty_like(x);

    bool bias_broadcast = (bias.numel() == 1);
    bool scale_broadcast = (scale.numel() == 1);

    float eps = 1e-5;

    shared_memory_coalesced_access_cuda(x, bias, scale, y, gn_weight, gn_bias,
                                        num_groups, bias_broadcast, scale_broadcast, eps);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Shared memory coalesced access kernel (CUDA)");
}
