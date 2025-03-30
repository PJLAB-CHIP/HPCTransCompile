#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor clusters2,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    bool is_training,
    int64_t cluster_size,
    int64_t feature_size
) {
  // Dimensions extraction
  auto B = x.size(0);
  auto N = x.size(1);
  auto D = x.size(2);

  // Flatten input
  x = x.reshape({B * N, D});

  // Modular computation blocks
  auto assignment = [&] {
    auto a = at::matmul(x, clusters);
    a = at::batch_norm(a, bn_weight, bn_bias, bn_mean, bn_var, is_training, 0.1, 1e-5, true);
    return at::softmax(a, 1).narrow(1, 0, cluster_size);
  }();

  // Assignment processing
  assignment = assignment.reshape({B, N, cluster_size});
  auto a_sum = assignment.sum(1, true);

  // Final VLAD computation
  auto a = a_sum * clusters2;
  assignment = assignment.transpose(1, 2);
  x = x.reshape({B, N, D});

  auto vlad = at::bmm(assignment, x).transpose(1, 2) - a;
  vlad = vlad / (vlad.norm(2, {1}, true) + 1e-12);
  vlad = vlad.reshape({B, D * cluster_size});
  return vlad / (vlad.norm(2, {1}, true) + 1e-12);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized NetVLAD with ghost clusters (CPU)");
}