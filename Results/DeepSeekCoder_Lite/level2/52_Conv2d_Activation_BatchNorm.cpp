#include <torch/extension.h>
#include <vector>
#include <math.h>
#include <omp.h>

// Define block sizes for experimentation
#define ACTIVATION_BLOCK_SIZE 512
#define BN_BLOCK_SIZE 256

// Helper functions for math operations

template <typename scalar_t>
inline scalar_t my_exp(scalar_t x);

template <>
inline float my_exp<float>(float x) { return expf(x); }

template <>
inline double my_exp<double>(double x) { return exp(x); }

template <typename scalar_t>
inline scalar_t my_log1p(scalar_t x);

template <>
inline float my_log1p<float>(float x) { return log1pf(x); }

template <>
inline double my_log1p<double>(double x) { return log1p(x); }

template <typename scalar_t>
inline scalar_t my_tanh(scalar_t x);

template <>
inline float my_tanh<float>(float x) { return tanhf(x); }

template <>
inline double my_tanh<double>(double x) { return tanh(x); }

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

  // Calculate per-channel sums and sums of squares
  std::vector<double> d_sum(C, 0.0);
  std::vector<double> d_sumsq(C, 0.0);

  #pragma omp parallel for
  for (int c = 0; c < C; ++c) {
      double local_sum = 0;
      double local_sumsq = 0;
      for (int n = 0; n < N; ++n) {
          for (int h = 0; h < H; ++h) {
              for (int w = 0; w < W; ++w) {
                  int i = n * (C * H * W) + c * (H * W) + h * W + w;
                  scalar_t val = x[i];
                  scalar_t sp = my_log1p<scalar_t>(my_exp<scalar_t>(val)); // softplus(x)
                  scalar_t th = my_tanh<scalar_t>(sp); // tanh(softplus(x))
                  scalar_t act = val * th;             // x * tanh(softplus(x))
                  local_sum += static_cast<double>(act);
                  local_sumsq += static_cast<double>(act * act);
              }
          }
      }
      d_sum[c] = local_sum;
      d_sumsq[c] = local_sumsq;
  }

  // Calculate mean and variance for each channel
  for (int c = 0; c < C; ++c) {
      double count = static_cast<double>(N * H * W);
      double mean = d_sum[c] / count;
      double var = d_sumsq[c] / count - mean * mean;
      #pragma omp parallel for
      for (int n = 0; n < N; ++n) {
          for (int h = 0; h < H; ++h) {
              for (int w = 0; w < W; ++w) {
                  int i = n * (C * H * W) + c * (H * W) + h * W + w;
                  activated[i] = static_cast<scalar_t>(bn_weight[c] * (static_cast<scalar_t>(x[i]) - static_cast<scalar_t>(mean)) / sqrt(var + static_cast<double>(eps)) + bn_bias[c]);
              }
          }
      }
  }

  return activated;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fused activation and batch normalization forward (CPU) with optimized block sizes");
}