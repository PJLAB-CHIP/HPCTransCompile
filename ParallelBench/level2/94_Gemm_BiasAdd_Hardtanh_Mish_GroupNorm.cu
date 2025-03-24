#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Fused kernel that performs BiasAdd, Hardtanh, Mish then GroupNorm normalization in one pass.
// Each block processes one group of channels for one sample.
// This version uses __ldg() for read-only global memory accesses on GEMM result and bias to achieve
// aligned 128-bit memory accesses and reduce latency.

template <typename scalar_t>
__global__ void fused_act_groupnorm_kernel(
    scalar_t* __restrict__ y,         // in/out tensor with shape [N, C]
    const scalar_t* __restrict__ bias,  // bias vector of length C
    const int N,
    const int C,
    const int num_groups,
    const float eps) {

  // Each block processes one sample and one group
  int sample = blockIdx.x; // sample index
  int group = blockIdx.y;  // group index
  int channels_per_group = C / num_groups;
  int group_start = group * channels_per_group;

  int tid = threadIdx.x;
  float act_val = 0.0f;  // activated value after bias, Hardtanh and Mish

  // Only threads with tid < channels_per_group are active
  if (tid < channels_per_group) {
    int channel = group_start + tid;
    int idx = sample * C + channel;
    // Use __ldg() for read-only global memory load of GEMM result and bias
    float tmp = static_cast<float>(__ldg(&y[idx])) + static_cast<float>(__ldg(&bias[channel]));
    // Hardtanh activation: clamp between -1 and 1
    tmp = fminf(fmaxf(tmp, -1.0f), 1.0f);
    // Mish activation: x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
    float sp = log1pf(expf(tmp));
    act_val = tmp * tanhf(sp);
  }

  // Allocate shared memory for reduction of sum and sum of squares
  // Shared memory layout: first blockDim.x floats for sum and next blockDim.x for sum of squares.
  extern __shared__ float shared_mem[];
  float* s_sum = shared_mem;
  float* s_sum_sq = shared_mem + blockDim.x;

  float temp = (tid < channels_per_group) ? act_val : 0.0f;
  s_sum[tid] = temp;
  s_sum_sq[tid] = temp * temp;
  __syncthreads();

  // Parallel reduction in shared memory to compute the sum and sum of squares
  int nthreads = blockDim.x;
  for (int stride = nthreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Compute mean and variance for the group
  float mean = s_sum[0] / channels_per_group;
  float variance = s_sum_sq[0] / channels_per_group - mean * mean;
  float inv_std = rsqrtf(variance + eps);
  __syncthreads();

  // Write the normalized result back to global memory
  if (tid < channels_per_group) {
    int channel = group_start + tid;
    int idx = sample * C + channel;
    float norm_val = (act_val - mean) * inv_std;
    y[idx] = static_cast<scalar_t>(norm_val);
  }
}

// Host function to launch the fused kernel
// It performs GEMM (with weight_bias addition), followed by a fused kernel that applies bias addition,
// Hardtanh, Mish, and GroupNorm in one pass using optimized global memory read via __ldg().

torch::Tensor fused_activation_groupnorm_cuda(
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
  block_size = min(block_size, 1024);

  // Grid dimensions: one block per sample per group
  dim3 grid(N, num_groups);
  dim3 block(block_size);

  // Dynamic shared memory size: two arrays of block_size floats
  size_t shared_mem_size = 2 * block_size * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "fused_activation_groupnorm_cuda", ([&] {
    fused_act_groupnorm_kernel<scalar_t><<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        N,
        C,
        num_groups,
        static_cast<float>(eps));
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
  y = fused_activation_groupnorm_cuda(y, bias, num_groups, eps);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fused BiasAdd, Hardtanh, Mish and GroupNorm CUDA forward function",
        py::arg("x"),
        py::arg("weight"),
        py::arg("weight_bias"),
        py::arg("bias"),
        py::arg("num_groups"),
        py::arg("eps") = 1e-5);
}
