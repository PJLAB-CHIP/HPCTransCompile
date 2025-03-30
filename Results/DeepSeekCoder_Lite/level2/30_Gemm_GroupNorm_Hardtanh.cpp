#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>

// Define tile size for matrix multiplication tiling
constexpr int TILE_SIZE = 16;

//-------------------------------------------------------------------
// Modular device functions for Linear Forward (GEMM) using tiling
//-------------------------------------------------------------------

// Load a tile from the input matrix (x) into shared memory
// x: [batch_size, in_features]
// Each block row corresponds to one row of x, load TILE_SIZE elements per iteration

template <typename scalar_t, int TILE_SIZE>
__inline__ void load_tile_A(const scalar_t* __restrict__ x,
                             scalar_t A_tile[TILE_SIZE][TILE_SIZE],
                             int row, int t, int in_features) {
  int tx = threadIdx.x;
  int col = t * TILE_SIZE + tx;
  A_tile[threadIdx.y][tx] = (col < in_features) ? x[row * in_features + col] : static_cast<scalar_t>(0);
}

// Load a tile from the weight matrix into shared memory
// weight: [out_features, in_features]
// For a given output feature (col), load TILE_SIZE elements from weight

template <typename scalar_t, int TILE_SIZE>
__inline__ void load_tile_B(const scalar_t* __restrict__ weight,
                             scalar_t B_tile[TILE_SIZE][TILE_SIZE],
                             int col, int t, int in_features) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int k = t * TILE_SIZE + ty;
  B_tile[ty][tx] = (k < in_features) ? weight[col * in_features + k] : static_cast<scalar_t>(0);
}

// Compute dot product on a single tile loaded into shared memory
// Multiplying the row of A_tile with the column of B_tile

template <typename scalar_t, int TILE_SIZE>
__inline__ scalar_t compute_tile_dot(scalar_t A_tile[TILE_SIZE][TILE_SIZE],
                                      scalar_t B_tile[TILE_SIZE][TILE_SIZE]) {
  scalar_t sum = 0;
  #pragma unroll
  for (int i = 0; i < TILE_SIZE; i++) {
    sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
  }
  return sum;
}

// Linear Forward Kernel using modular device functions and shared memory tiling

template <typename scalar_t, int TILE_SIZE>
__global__ void linear_forward_kernel_modular(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  scalar_t sum = 0;
  int numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

  __shared__ scalar_t A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ scalar_t B_tile[TILE_SIZE][TILE_SIZE];

  for (int t = 0; t < numTiles; t++) {
    load_tile_A<scalar_t, TILE_SIZE>(x, A_tile, row, t, in_features);
    load_tile_B<scalar_t, TILE_SIZE>(weight, B_tile, col, t, in_features);
    __syncthreads();
    sum += compute_tile_dot<scalar_t, TILE_SIZE>(A_tile, B_tile);
    __syncthreads();
  }

  if (row < batch_size && col < out_features) {
    output[row * out_features + col] = sum + bias[col];
  }
}

//-------------------------------------------------------------------
// Modular device functions for Group Normalization with parallel reduction
//-------------------------------------------------------------------

// A simple block-wide reduction to sum up values in shared memory

template <typename scalar_t>
__inline__ scalar_t blockReduceSum(volatile scalar_t* sdata, int tid, int blockDim) {
  for (int s = blockDim / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  return sdata[0];
}

// Group Normalization Kernel: Each block handles one (batch, group) pair

template <typename scalar_t>
__global__ void group_norm_forward_kernel_modular(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gamma,  // scale parameter
    const scalar_t* __restrict__ beta,   // shift parameter
    scalar_t* __restrict__ output,
    int batch_size,
    int num_channels,
    int num_groups) {
  int channels_per_group = num_channels / num_groups;
  int idx = blockIdx.x; // total blocks = batch_size * num_groups
  int batch = idx / num_groups;
  int group = idx % num_groups;

  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

  // Compute mean in parallel over channels in the group
  scalar_t sum = 0;
  for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
    int channel = group * channels_per_group + i;
    sum += x[batch * num_channels + channel];
  }
  sdata[threadIdx.x] = sum;
  __syncthreads();

  scalar_t mean = blockReduceSum<scalar_t>(sdata, threadIdx.x, blockDim.x) / channels_per_group;
  __syncthreads();

  // Compute variance in parallel
  scalar_t sq_sum = 0;
  for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
    int channel = group * channels_per_group + i;
    scalar_t diff = x[batch * num_channels + channel] - mean;
    sq_sum += diff * diff;
  }
  sdata[threadIdx.x] = sq_sum;
  __syncthreads();

  scalar_t var = blockReduceSum<scalar_t>(sdata, threadIdx.x, blockDim.x) / channels_per_group;
  __syncthreads();

  scalar_t inv_std = rsqrtf(var + 1e-5f);

  // Normalize, scale, and shift each feature in this group
  for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
    int channel = group * channels_per_group + i;
    scalar_t val = x[batch * num_channels + channel];
    output[batch * num_channels + channel] = ((val - mean) * inv_std) * gamma[channel] + beta[channel];
  }
}

//-------------------------------------------------------------------
// Modular device function for Hardtanh Activation
//-------------------------------------------------------------------

template <typename scalar_t>
__inline__ scalar_t hardtanh_activation(scalar_t val, scalar_t min_val, scalar_t max_val) {
  return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

// Hardtanh Kernel: Applies the activation in a grid-stride loop

template <typename scalar_t>
__global__ void hardtanh_forward_kernel_modular(
    const scalar_t* __restrict__ x,
    scalar_t min_val,
    scalar_t max_val,
    scalar_t* __restrict__ output,
    size_t total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; idx < total_elements; idx += stride) {
    scalar_t val = x[idx];
    output[idx] = hardtanh_activation<scalar_t>(val, min_val, max_val);
  }
}

//-------------------------------------------------------------------
// Host functions launching the kernels
//-------------------------------------------------------------------

void linear_forward_cuda_modular(
    at::Tensor x, 
    at::Tensor weight, 
    at::Tensor bias, 
    at::Tensor output) {

  const int batch_size = x.size(0);
  const int in_features = x.size(1);
  const int out_features = weight.size(0);

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_features + TILE_SIZE - 1) / TILE_SIZE,
            (batch_size + TILE_SIZE - 1) / TILE_SIZE);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "linear_forward_cuda_modular", ([&] {
    linear_forward_kernel_modular<scalar_t, TILE_SIZE><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        in_features,
        out_features);
  }));
}

void group_norm_forward_cuda_modular(
    at::Tensor x, 
    at::Tensor gamma,  // Group norm weight
    at::Tensor beta,   // Group norm bias
    int64_t num_groups,
    at::Tensor output) {

  const int batch_size = x.size(0);
  const int num_channels = x.size(1);
  int total_blocks = batch_size * num_groups;
  int channels_per_group = num_channels / num_groups;
  int threads = (channels_per_group < 256) ? channels_per_group : 256;
  size_t shared_mem = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda_modular", ([&] {
    group_norm_forward_kernel_modular<scalar_t><<<total_blocks, threads, shared_mem>>>(
        x.data_ptr<scalar_t>(),
        gamma.data_ptr<scalar_t>(),
        beta.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        num_channels,
        num_groups);
  }));
}

void hardtanh_forward_cuda_modular(
    at::Tensor x, 
    float min_val, 
    float max_val,
    at::Tensor output) {

  const size_t total_elements = x.numel();
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_forward_cuda_modular", ([&] {
    hardtanh_forward_kernel_modular<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        static_cast<scalar_t>(min_val),
        static_cast<scalar_t>(max_val),
        output.data_ptr<scalar_t>(),
        total_elements);
  }));
}

//-------------------------------------------------------------------
// Combined Host Function: Executes linear, group norm, and hardtanh sequentially
//-------------------------------------------------------------------

at::Tensor module_fn_cuda_forward(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor group_norm_weight,
    at::Tensor group_norm_bias,
    int64_t num_groups,
    float hardtanh_min,
    float hardtanh_max) {

  // Ensure inputs are contiguous and on CUDA
  x = x.contiguous();
  weight = weight.contiguous();
  bias = bias.contiguous();
  group_norm_weight = group_norm_weight.contiguous();
  group_norm_bias = group_norm_bias.contiguous();

  int64_t batch_size = x.size(0);
  int64_t in_features = x.size(1);
  int64_t out_features = weight.size(0);

  auto options = x.options();
  at::Tensor linear_output = at::empty({batch_size, out_features}, options);
  at::Tensor group_norm_output = at::empty({batch_size, out_features}, options);
  at::Tensor output = at::empty({batch_size, out_features}, options);

  // Linear layer computation with tiling
  linear_forward_cuda_modular(x, weight, bias, linear_output);

  // Group Normalization with parallel reduction per group
  group_norm_forward_cuda_modular(linear_output, group_norm_weight, group_norm_bias, num_groups, group_norm_output);

  // Hardtanh activation using a grid-stride loop
  hardtanh_forward_cuda_modular(group_norm_output, hardtanh_min, hardtanh_max, output);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_cuda_forward, "Forward pass (CUDA modular optimized)");
}