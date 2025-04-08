#include <torch/extension.h>
#include <omp.h>

// Function to perform matrix multiplication
template <typename scalar_t>
void matmul(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
  auto a_data = a.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  auto c_data = c.data_ptr<scalar_t>();

  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b.size(1);

  #pragma omp parallel for collapse(2)
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      scalar_t sum = 0;
      for (int64_t l = 0; l < k; ++l) {
        sum += a_data[i * k + l] * b_data[l * n + j];
      }
      c_data[i * n + j] = sum;
    }
  }
}

// Function to perform element-wise addition
template <typename scalar_t>
void add(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
  auto a_data = a.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  auto c_data = c.data_ptr<scalar_t>();

  int64_t size = a.numel();

  #pragma omp parallel for
  for (int64_t i = 0; i < size; ++i) {
    c_data[i] = a_data[i] + b_data[i];
  }
}

// Function to apply ReLU activation
template <typename scalar_t>
void relu(torch::Tensor& x) {
  auto x_data = x.data_ptr<scalar_t>();
  int64_t size = x.numel();

  #pragma omp parallel for
  for (int64_t i = 0; i < size; ++i) {
    if (x_data[i] < 0) {
      x_data[i] = 0;
    }
  }
}

// Forward function for the MLP
torch::Tensor forward(
    torch::Tensor x,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {

  for (size_t i = 0; i < weights.size() - 1; ++i) {
    // Perform linear transformation: x = x * weights[i] + biases[i]
    matmul<scalar_t>(x, weights[i], x);
    add<scalar_t>(x, biases[i], x);
    // Apply ReLU activation
    relu<scalar_t>(x);
  }
  // Final linear transformation
  matmul<scalar_t>(x, weights.back(), x);
  add<scalar_t>(x, biases.back(), x);
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "MLP forward");
}