#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

// Define block sizes for experimentation
#define ACTIVATION_BLOCK_SIZE 512
#define BN_BLOCK_SIZE 256

// Helper functions for math operations

template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) { return expf(x); }

template <>
__device__ inline double my_exp<double>(double x) { return exp(x); }


template <typename scalar_t>
__device__ inline scalar_t my_log1p(scalar_t x);

template <>
__device__ inline float my_log1p<float>(float x) { return log1pf(x); }

template <>
__device__ inline double my_log1p<double>(double x) { return log1p(x); }


template <typename scalar_t>
__device__ inline scalar_t my_tanh(scalar_t x);

template <>
__device__ inline float my_tanh<float>(float x) { return tanhf(x); }

template <>
__device__ inline double my_tanh<double>(double x) { return tanh(x); }


// Kernel 1: Fused activation and per-channel reduction
// Activation: act = x * tanh( softplus(x) ) with softplus(x) = log1p(exp(x))
// Each block processes one channel, using grid-stride loops to cover all elements in a channel.

template <typename scalar_t>
__global__ void activation_reduction_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int N, int C, int H, int W,
    scalar_t* __restrict__ d_sum,
    scalar_t* __restrict__ d_sumsq) {

  // Each block is assigned to one channel via blockIdx.y
  int c = blockIdx.y;
  int count = N * H * W;  // number of elements per channel
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_t local_sum = 0;
  scalar_t local_sumsq = 0;

  // Process elements for the channel using grid-stride loop
  for (; i < count; i += blockDim.x * gridDim.x) {
      int HW = H * W;
      int n = i / HW;
      int rem = i % HW;
      int h = rem / W;
      int w = rem % W;
      int offset = n * (C * H * W) + c * (HW) + h * W + w;
      scalar_t val = x[offset];
      scalar_t sp = my_log1p<scalar_t>(my_exp<scalar_t>(val)); // softplus(x)
      scalar_t th = my_tanh<scalar_t>(sp); // tanh(softplus(x))
      scalar_t act = val * th;             // x * tanh(softplus(x))
      y[offset] = act;
      local_sum += act;
      local_sumsq += act * act;
  }

  // Warp-level reduction
  unsigned int lane = threadIdx.x & 31;
  for (int offset = 16; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, offset);
  }

  // Shared memory for partial warp reductions
  __shared__ scalar_t warpSum[32];
  __shared__ scalar_t warpSumSq[32];

  int warp_id = threadIdx.x / 32;
  if (lane == 0) {
      warpSum[warp_id] = local_sum;
      warpSumSq[warp_id] = local_sumsq;
  }
  __syncthreads();

  // First warp reduces the partial sums
  local_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? warpSum[lane] : 0;
  local_sumsq = (threadIdx.x < (blockDim.x + 31) / 32) ? warpSumSq[lane] : 0;
  if (threadIdx.x < 32) {
      for (int offset = 16; offset > 0; offset /= 2) {
          local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
          local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, offset);
      }
      if (threadIdx.x == 0) {
          atomicAdd(&d_sum[c], local_sum);
          atomicAdd(&d_sumsq[c], local_sumsq);
      }
  }
}

// Kernel 2: Batch Normalization
// Uses computed per-channel sums to calculate mean and variance, then normalizes and applies affine transformation.

template <typename scalar_t>
__global__ void batchnorm_kernel(
    scalar_t* __restrict__ y,
    int N, int C, int H, int W,
    const scalar_t* __restrict__ d_sum,
    const scalar_t* __restrict__ d_sumsq,
    const scalar_t* __restrict__ bn_weight,
    const scalar_t* __restrict__ bn_bias,
    scalar_t eps) {

  int total = N * C * H * W;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
      int w = i % W;
      int h = (i / W) % H;
      int c = (i / (W * H)) % C;
      // Number of elements per channel
      scalar_t count = static_cast<scalar_t>(N * H * W);
      scalar_t mean = d_sum[c] / count;
      scalar_t var = d_sumsq[c] / count - mean * mean;
      scalar_t norm = (y[i] - mean) / sqrt(var + eps);
      y[i] = bn_weight[c] * norm + bn_bias[c];
  }
}

// Forward function: performs convolution, then fused activation with reduction, and finally batch normalization.

torch::Tensor forward(
    torch::Tensor x,
    double eps,
    double momentum,  // momentum is not used in fused computation; training mode is assumed
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,  // not used in fused BN
    torch::Tensor bn_running_var) {  // not used in fused BN

  // Convolution
  x = torch::conv2d(x, conv_weight, conv_bias);

  auto activated = torch::empty_like(x);

  int N = x.size(0);
  int C = x.size(1);
  int H = x.size(2);
  int W = x.size(3);
  int count = N * H * W; // Elements per channel

  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  auto d_sum = torch::zeros({C}, options);
  auto d_sumsq = torch::zeros({C}, options);

  // Launch fused activation and reduction kernel with tuned block size
  int act_blocks = (count + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE;
  dim3 act_grid(act_blocks, C);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "activation_reduction_kernel", ([&] {
      activation_reduction_kernel<scalar_t><<<act_grid, ACTIVATION_BLOCK_SIZE>>>(
          x.data_ptr<scalar_t>(),
          activated.data_ptr<scalar_t>(),
          N, C, H, W,
          d_sum.data_ptr<scalar_t>(),
          d_sumsq.data_ptr<scalar_t>());
  }));

  // Launch batch normalization kernel with tuned block size
  int total = activated.numel();
  int bn_blocks = (total + BN_BLOCK_SIZE - 1) / BN_BLOCK_SIZE;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "batchnorm_kernel", ([&] {
      batchnorm_kernel<scalar_t><<<bn_blocks, BN_BLOCK_SIZE>>>(
          activated.data_ptr<scalar_t>(),
          N, C, H, W,
          d_sum.data_ptr<scalar_t>(),
          d_sumsq.data_ptr<scalar_t>(),
          bn_weight.data_ptr<scalar_t>(),
          bn_bias.data_ptr<scalar_t>(),
          static_cast<scalar_t>(eps));
  }));

  return activated;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fused activation and batch normalization forward (CUDA) with optimized block sizes");
}
