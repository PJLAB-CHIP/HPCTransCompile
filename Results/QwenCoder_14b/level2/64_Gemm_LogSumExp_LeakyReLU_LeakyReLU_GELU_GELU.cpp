#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template <typename scalar_t>
void tiled_linear_cpu(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    int M, int K, int N) {
  #pragma omp parallel for collapse(2)
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      scalar_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += x[row * K + k] * weight[col * K + k];
      }
      y[row * N + col] = sum + bias[col];
    }
  }
}

template <typename scalar_t>
void tiled_logsumexp_cpu(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int width) {
  #pragma omp parallel for
  for (int row = 0; row < width; ++row) {
    scalar_t local_max = -INFINITY;
    for (int i = 0; i < width; ++i) {
      local_max = fmax(local_max, input[row * width + i]);
    }
    scalar_t local_sum = 0;
    for (int i = 0; i < width; ++i) {
      local_sum += exp(input[row * width + i] - local_max);
    }
    output[row] = local_max + log(local_sum);
  }
}

template <typename scalar_t>
void fused_leaky_relu_cpu(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    scalar_t negative_slope,
    int size) {
  #pragma omp parallel for
  for (int idx = 0; idx < size; ++idx) {
    scalar_t val = x[idx];
    scalar_t mini = (val - fabs(val)) * static_cast<scalar_t>(0.5);
    y[idx] = val + (negative_slope * negative_slope - static_cast<scalar_t>(1)) * mini;
  }
}

template <typename scalar_t>
void fused_gelu_cpu(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int size) {
  #pragma omp parallel for
  for (int idx = 0; idx < size; ++idx) {
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

void linear_forward_cpu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor y) {
  int batch_size = x.size(0);
  int in_features = x.size(1);
  int out_features = weight.size(0);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tiled_linear_forward_cpu", ([&] {
    tiled_linear_cpu<scalar_t>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        batch_size, in_features, out_features);
  }));
}

void logsumexp_forward_cpu(
    torch::Tensor x,
    torch::Tensor y) {
  int batch_size = x.size(0);
  int width = x.size(1);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tiled_logsumexp_forward_cpu", ([&] {
    tiled_logsumexp_cpu<scalar_t>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        width);
  }));
}

void fused_leaky_relu_forward_cpu(
    torch::Tensor x,
    torch::Tensor y,
    float negative_slope) {
  int size = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_leaky_relu_forward_cpu", ([&] {
    fused_leaky_relu_cpu<scalar_t>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        static_cast<scalar_t>(negative_slope),
        size);
  }));
}

void fused_gelu_forward_cpu(
    torch::Tensor x,
    torch::Tensor y) {
  int size = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_gelu_forward_cpu", ([&] {
    fused_gelu_cpu<scalar_t>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        size);
  }));
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias) {
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(bias);

  auto batch_size = x.size(0);
  auto in_features = x.size(1);
  auto out_features = weight.size(0);

  auto options = x.options();
  // Compute linear transformation: y_linear = x @ weight^T + bias
  auto y_linear = torch::empty({batch_size, out_features}, options);
  linear_forward_cpu(x, weight, bias, y_linear);

  // Compute LogSumExp across dim=1 (each row)
  auto y_logsumexp = torch::empty({batch_size, 1}, options);
  logsumexp_forward_cpu(y_linear, y_logsumexp);

  // Fuse two consecutive LeakyReLU activations into one kernel call
  auto y_leaky = torch::empty_like(y_logsumexp);
  fused_leaky_relu_forward_cpu(y_logsumexp, y_leaky, 0.01f);

  // Fuse two consecutive GELU activations into one kernel call
  auto y_gelu = torch::empty_like(y_leaky);
  fused_gelu_forward_cpu(y_leaky, y_gelu);

  return y_gelu;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Module function forward");
}