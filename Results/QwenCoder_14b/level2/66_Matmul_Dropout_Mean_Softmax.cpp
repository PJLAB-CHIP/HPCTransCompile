#include <torch/extension.h>
#include <omp.h>
#include <chrono>

void forward_cpu(
    const float* x, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    float dropout_p, bool training, unsigned long long seed) {

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
        unsigned long long offset = i * out_features + j;
        unsigned int xorshifted = ((offset >> 18) ^ offset) * 0x2545F49B2D787CF5ULL;
        xorshifted ^= xorshifted >> 27;
        xorshifted *= 0x2545F49B2D787CF5ULL;
        xorshifted ^= xorshifted >> 27;
        float rand_val = xorshifted / static_cast<float>(UINT_MAX);
        if (rand_val < dropout_p) {
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

torch::Tensor forward_cpu_wrapper(
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

  // Call CPU function
  forward_cpu(
    x_data, weight_data, bias_data,
    output_data,
    batch_size, in_features, out_features,
    dropout_p, training, seed
  );

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cpu_wrapper, "Custom forward CPU implementation");
}