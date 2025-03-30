#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {

  // Ensure the inputs are correct
  TORCH_CHECK(weights.size() == biases.size(), "weights and biases must have the same size");
  TORCH_CHECK(weights.size() > 0, "weights and biases must have at least one element");

  // Perform the same operations as the CUDA kernel
  for (size_t i = 0; i < weights.size() - 1; ++i) {
    x = torch::addmm(biases[i], x, weights[i].transpose(0, 1)).relu();
  }
  x = torch::addmm(biases.back(), x, weights.back().transpose(0, 1));
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "MLP forward");
}