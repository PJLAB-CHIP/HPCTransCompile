#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Fused kernel that performs BiasAdd, Hardtanh, Mish then GroupNorm normalization in one pass.
// Each block processes one group of channels for one sample.
// This version uses __ldg() for read-only global memory accesses on GEMM result and bias to achieve
// aligned 128-bit memory accesses and reduce latency.

template <typename scalar_t>
void fused_act_groupnorm_kernel(
    scalar_t* __restrict__ y,         // in/out tensor with shape [N, C]
    const scalar_t* __restrict__ bias,  // bias vector of length C
    const int N,
    const int C,
    const int num_groups,
    const float eps,
    const int sample,
    const int group,
    const int channels_per_group) {

  int tid = omp_get_thread_num();
  float act_val = 0.0f;  // activated value after bias, Hardtanh and Mish

  // Only threads with tid < channels_per_group are active
  if (tid < channels_per_group) {
    int channel = group * channels_per_group + tid;
    int idx = sample * C + channel;
    // Use __ldg() for read-only global memory load of GEMM result and bias
    float tmp = static_cast<float>(y[idx]) + static_cast<float>(bias[channel]);
    // Hardtanh activation: clamp between -1 and 1
    tmp = fminf(fmaxf(tmp, -1.0f), 1.0f);
    // Mish activation: x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
    float sp = log1pf(expf(tmp));
    act_val = tmp * tanhf(sp);

    // Allocate shared memory for reduction of sum and sum of squares
    // Shared memory layout: first blockDim.x floats for sum and next blockDim.x for sum of squares.
    extern float* shared_mem;
    float* s_sum = shared_mem;
    float* s_sum_sq = shared_mem + omp_get_max_threads();

    s_sum[tid] = act_val;
    s_sum_sq[tid] = act_val * act_val;
    __sync_sub_and_fetch(&s_sum[tid], 0);
    __sync_sub_and_fetch(&s_sum_sq[tid], 0);

    // Parallel reduction in shared memory to compute the sum and sum of squares
    for (int stride = omp_get_max_threads() / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_sum[tid] += s_sum[tid + stride];
        s_sum_sq[tid] += s_sum_sq[tid + stride];
      }
    }

    // Compute mean and variance for the group
    float mean = s_sum[0] / channels_per_group;
    float variance = s_sum_sq[0] / channels_per_group - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    // Write the normalized result back to global memory
    if (tid < channels_per_group) {
      int idx = sample * C + channel;
      float norm_val = (act_val - mean) * inv_std;
      y[idx] = static_cast<scalar_t>(norm_val);
    }
  }
}

// Host function to launch the fused kernel
// It performs GEMM (with weight_bias addition), followed by a fused kernel that applies bias addition,
// Hardtanh, Mish, and GroupNorm in one pass using optimized global memory read via __ldg().

torch::Tensor fused_activation_groupnorm_cpu(
    torch::Tensor y,
    torch::Tensor bias,
    int num_groups,
    double eps) {
  CHECK_INPUT(y);
  CHECK_INPUT(bias);
  TORCH_CHECK(y.dim() == 2, "Input tensor y must be 2D");
  int N = y.size(0);
  int C = y.size(1);
  TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");
  int channels_per_group = C / num_groups;

  // Determine block size as the next multiple of 32 (warp size) that can accommodate channels_per_group, capped at 1024
  int block_size = ((channels_per_group + 31) / 32) * 32;
  block_size = std::min(block_size, 1024);

  // Grid dimensions: one block per sample per group
  int num_blocks = N * num_groups;

  // Dynamic shared memory size: two arrays of block_size floats
  size_t shared_mem_size = 2 * block_size * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "fused_activation_groupnorm_cpu", ([&] {
    float* y_data = y.data_ptr<scalar_t>();
    const float* bias_data = bias.data_ptr<scalar_t>();

    #pragma omp parallel for schedule(dynamic)
    for (int sample = 0; sample < N; ++sample) {
      for (int group = 0; group < num_groups; ++group) {
        fused_act_groupnorm_kernel<scalar_t>(
            y_data + sample * C + group * channels_per_group,
            bias_data + group * channels_per_group,
            N,
            C,
            num_groups,
            static_cast<float>(eps),
            sample,
            group,
            channels_per_group);
      }
    }
  }));

  return y;
}

// The forward function performs GEMM (with an added weight_bias) followed by the fused kernel.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor weight_bias,
    torch::Tensor bias,
    int64_t num_groups,
    double eps = 1e-5) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(weight_bias);
  CHECK_INPUT(bias);

  // GEMM: x @ weight.t() + weight_bias
  auto y = torch::matmul(x, weight.t()) + weight_bias;

  // Fuse second bias addition, Hardtanh, Mish, and GroupNorm into a single kernel
  y = fused_activation_groupnorm_cpu(y, bias, num_groups, eps);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fused BiasAdd, Hardtanh, Mish and GroupNorm CPU forward function",
        py::arg("x"),
        py::arg("weight"),
        py::arg("weight_bias"),
        py::arg("bias"),
        py::arg("num_groups"),
        py::arg("eps") = 1e-5);
}