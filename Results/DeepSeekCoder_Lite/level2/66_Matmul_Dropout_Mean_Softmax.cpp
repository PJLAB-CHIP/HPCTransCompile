#include <torch/extension.h>
#include <random>
#include <chrono>
#include <omp.h>

void forward_kernel(
    const float* x, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    float dropout_p, bool training, unsigned long long seed) {

  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  #pragma omp parallel for
  for (int i = 0; i < batch_size; i++) {
    float sum_total = 0.0f;

    for (int j = 0; j < out_features; j++) {
      // Compute linear transformation: x[i] * weight[j] + bias[j]
      float linear_val = 0.0f;
      for (int k = 0; k < in_features; k++) {
        linear_val += x[i * in_features + k] * weight[j * in_features + k];
      }
      linear_val += bias[j];

      // Apply dropout
      if (training) {
        if (distribution(generator) < dropout_p) {
          linear_val = 0.0f;
        } else {
          linear_val /= (1.0f - dropout_p);
        }
      }

      sum_total += linear_val;
    }

    // Softmax on single element is always 1.0
    output[i] = 1.0f;
  }
}

torch::Tensor forward_cpu(
    torch::Tensor x,
    float dropout_p,
    bool training,
    torch::Tensor weight,
    torch::Tensor bias) {

  int batch_size = x.size(0);
  int in_features = x.size(1);
  int out_features = weight.size(0);

  auto output = torch::empty({batch_size, 1}, x.options());

  // Get data pointers
  auto x_data = x.data_ptr<float>();
  auto weight_data = weight.data_ptr<float>();
  auto bias_data = bias.data_ptr<float>();
  auto output_data = output.data_ptr<float>();

  // Generate random seed (use PyTorch's generator in production)
  unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

  // Launch kernel
  forward_kernel(
    x_data, weight_data, bias_data,
    output_data,
    batch_size, in_features, out_features,
    dropout_p, training, seed
  );

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cpu, "Custom forward CPU implementation");
}