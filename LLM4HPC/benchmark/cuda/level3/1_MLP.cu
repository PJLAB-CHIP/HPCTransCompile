#include <torch/extension.h>

torch::Tensor forward(
    torch::Tensor x,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {

  for (size_t i = 0; i < weights.size() - 1; ++i) {
    x = torch::linear(x, weights[i], biases[i]);
    x = torch::relu(x);
  }
  x = torch::linear(x, weights.back(), biases.back());
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "MLP forward");
}