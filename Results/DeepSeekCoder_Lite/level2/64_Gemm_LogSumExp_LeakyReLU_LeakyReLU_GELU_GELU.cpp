#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// Tiled GEMM parameters
#define TILE_DIM 16

// Tiled linear kernel using shared memory
// Computes: y = x * (weight)^T + bias
// x: [M x K]
// weight: [N x K]  (each row corresponds to an output feature), used as weight^T
// y: [M x N] where M = batch_size, K = in_features, N = out_features

template <typename scalar_t>
void tiled_linear_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    int M, int K, int N, int tile_dim) {
  // Calculate row and column indices of C (output)
  int row, col;
  #pragma omp parallel for private(row, col) schedule(dynamic)
  for (row = 0; row < M; ++row) {
    for (col = 0; col < N; ++col) {
      scalar_t sum = 0;

      // Declare shared memory for tiles from x and weight
      __restrict__ scalar_t sA[tile_dim][tile_dim];
      __restrict__ scalar_t sB[tile_dim][tile_dim];

      // Loop over tiles
      for (int t = 0; t < (K + tile_dim - 1) / tile_dim; ++t) {
        // Load tile from x (matrix A)
        for (int i = 0; i < tile_dim; ++i) {
          for (int j = 0; j < tile_dim; ++j) {
            int x_idx = (row * K) + (t * tile_dim) + (i * K + j);
            sA[i][j] = (x_idx < (row * K + K)) ? x[x_idx] : static_cast<scalar_t>(0);
          }
        }

        // Load tile from weight^T. Note: weight is stored as [N x K], so weight[col][k] = weight[col * K + k].
        for (int i = 0; i < tile_dim; ++i) {
          for (int j = 0; j < tile_dim; ++j) {
            int weight_idx = (col * K) + (t * tile_dim) + (i * K + j);
            sB[i][j] = (weight_idx < (col * K + K)) ? weight[weight_idx] : static_cast<scalar_t>(0);
          }
        }

        // Compute partial dot product for the tile
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int i = 0; i < tile_dim; ++i) {
          for (int j = 0; j < tile_dim; ++j) {
            for (int k = 0; k < tile_dim; ++k) {
              sum += sA[i][k] * sB[k][j];
            }
          }
        }
      }

      // Write result with bias addition
      if (row < M && col < N) {
        y[(row * N) + col] = sum + bias[col];
      }
    }
  }
}

// Tiled LogSumExp kernel with parallel reduction
// Each block processes one row of the input matrix, reducing over 'width' elements

template <typename scalar_t>
void tiled_logsumexp_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int width, int tile_dim) {
  // Each block handles one row
  int row;
  #pragma omp parallel for private(row) schedule(dynamic)
  for (row = 0; row < width; ++row) {
    __restrict__ scalar_t sdata[256]; // shared memory for reduction - fixed size for 256 threads

    // Compute local maximum for the row
    scalar_t local_max = -INFINITY;
    for (int i = 0; i < width; i += 256) {
      scalar_t val = input[row * width + i];
      local_max = std::fmax(local_max, val);
    }
    sdata[threadIdx.x] = local_max;
    __syncwarp();

    // Reduce to find the maximum value in the row
    for (int s = 256 / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sdata[threadIdx.x] = std::fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
      }
      __syncwarp();
    }
    scalar_t max_val = sdata[0];

    // Compute local sum of exp(x - max_val)
    scalar_t local_sum = 0;
    for (int i = 0; i < width; i += 256) {
      scalar_t val = input[row * width + i];
      local_sum += std::exp(val - max_val);
    }
    sdata[threadIdx.x] = local_sum;
    __syncwarp();

    // Reduce to sum all local sums
    for (int s = 256 / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
      }
      __syncwarp();
    }

    // Write the LogSumExp result
    if (threadIdx.x == 0) {
      output[row] = max_val + std::log(sdata[0]);
    }
  }
}

// Fused double LeakyReLU kernel (elementwise)
// Applies LeakyReLU twice in one pass
// For x >= 0, output remains x; for x < 0, output becomes (negative_slope^2)*x
// Uses branchless computation by computing 0.5*(x - fabs(x)) which equals min(x, 0).

template <typename scalar_t>
void fused_leaky_relu_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    scalar_t negative_slope,
    int size) {
  int idx;
  #pragma omp parallel for private(idx) schedule(static)
  for (idx = 0; idx < size; ++idx) {
    scalar_t val = x[idx];
    scalar_t mini = 0.5 * (val - std::fabs(val));
    y[idx] = val + (negative_slope * negative_slope - static_cast<scalar_t>(1)) * mini;
  }
}

// Fused double GELU kernel (elementwise)
// Applies the GELU activation function twice
// GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

template <typename scalar_t>
void fused_gelu_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int size) {
  int idx;
  #pragma omp parallel for private(idx) schedule(static)
  for (idx = 0; idx < size; ++idx) {
    scalar_t v = x[idx];
    scalar_t k0 = static_cast<scalar_t>(0.5);
    scalar_t k1 = static_cast<scalar_t>(1.0);
    scalar_t sqrt_2_over_pi = static_cast<scalar_t>(0.7978845608);  // sqrt(2/pi)
    scalar_t cdf = k0 * (k1 + tanh(sqrt_2_over_pi * (v + static_cast<scalar_t>(0.044715) * v * v * v)));
    v = v * cdf;
    cdf = k0 * (k1 + tanh(sqrt_2_over_pi * (v + static_cast<scalar_t>(0.044715) * v * v * v)));
    y[idx] = v * cdf;
  }
}

// Linear forward CUDA function using the tiled GEMM kernel
void linear_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor y) {
  int batch_size = x.size(0);
  int in_features = x.size(1);
  int out_features = weight.size(0);

  const int TILE = TILE_DIM;
  int tile_dim = TILE;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tiled_linear_forward_cuda", ([&] {
    tiled_linear_kernel<scalar_t>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        batch_size, in_features, out_features, tile_dim);
  }));
}

// LogSumExp forward CUDA function using the tiled reduction kernel
void logsumexp_forward_cuda(
    torch::Tensor x,
    torch::Tensor y) {
  int batch_size = x.size(0);
  int width = x.size(1);

  const int threads = 256;
  // One block per row; allocate shared memory of size 'threads * sizeof(scalar_t)'
  int tile_dim = TILE_DIM;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tiled_logsumexp_forward_cuda", ([&] {
    tiled_logsumexp_kernel<scalar_t>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        width, tile_dim);
  }));
}

// Fused LeakyReLU forward CUDA function (elementwise)
void fused_leaky_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor y,
    float negative_slope) {
  int size = x.numel();
  const int threads = 256;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_leaky_relu_forward_cuda", ([&] {
    fused_leaky_relu_kernel<scalar_t>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        static_cast<scalar_t>(negative_slope),
        size);
  }));
}

// Fused GELU forward CUDA function (elementwise)
void fused_gelu_forward_cuda(
    torch::Tensor x,
    torch::Tensor y) {
  int size = x.numel();
  const int threads = 256;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_gelu_forward_cuda", ([&] {
    fused_gelu_kernel<scalar_t>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        size);
  }));
}

// Main module function that chains the operations: linear -> logsumexp -> fused LeakyReLU -> fused GELU
torch::Tensor module_fn_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  auto batch_size = x.size(0);
  auto in_features = x.size(1);
  auto out_features = weight.size(0);

  auto options = x.options();
  // Compute linear transformation: y_linear = x @ weight^T + bias
  auto y_linear = torch::empty({batch_size, out_features}, options);
  linear_forward_cuda(x, weight, bias, y_linear);

  // Compute LogSumExp across dim=1 (each row)
  auto y_logsumexp = torch::empty({batch_size, 1}, options);
  logsumexp_forward_cuda(y_linear, y_logsumexp);

  // Fuse two consecutive LeakyReLU activations into one kernel call
  auto y_leaky = torch::empty_like(y_logsumexp);
  fused_leaky_relu_forward_cuda(y_logsumexp, y_leaky, 0.01f);

  // Fuse two consecutive GELU activations into one kernel call
  auto y_gelu = torch::empty_like(y_leaky);
  fused_gelu_forward_cuda(y_leaky, y_gelu);

  return y_gelu;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Module function forward");
}