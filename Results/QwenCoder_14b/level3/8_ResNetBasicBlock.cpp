#include <torch/extension.h>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv1_weight,
    torch::Tensor conv2_weight,
    torch::Tensor bn1_weight,
    torch::Tensor bn1_bias,
    torch::Tensor bn1_mean,
    torch::Tensor bn1_var,
    torch::Tensor bn2_weight,
    torch::Tensor bn2_bias,
    torch::Tensor bn2_mean,
    torch::Tensor bn2_var,
    torch::Tensor downsample_conv_weight,
    torch::Tensor downsample_bn_weight,
    torch::Tensor downsample_bn_bias,
    torch::Tensor downsample_bn_mean,
    torch::Tensor downsample_bn_var,
    int64_t stride,
    bool is_training) {

  // Save identity for residual connection
  auto identity = x.clone();

  // First conv block
  auto out = torch::conv2d(x, conv1_weight, {}, {stride, stride}, {1, 1});
  out = torch::batch_norm(out, bn1_weight, bn1_bias, bn1_mean, bn1_var, is_training, 0.1, 1e-5, true);
  out = torch::relu(out);

  // Second conv block
  out = torch::conv2d(out, conv2_weight, {}, {1, 1}, {1, 1});
  out = torch::batch_norm(out, bn2_weight, bn2_bias, bn2_mean, bn2_var, is_training, 0.1, 1e-5, true);

  // Downsample path - explicit IntArrayRef for padding
  identity = torch::conv2d(
      identity,
      downsample_conv_weight,
      {},
      {stride, stride},
      c10::IntArrayRef({0, 0})  // Explicit type disambiguation
  );
  identity = torch::batch_norm(
      identity,
      downsample_bn_weight,
      downsample_bn_bias,
      downsample_bn_mean,
      downsample_bn_var,
      is_training,
      0.1,
      1e-5,
      true
  );

  // Add and final activation
  out += identity;
  out = torch::relu(out);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "BasicBlock forward");
}
```
</cpu> ```cpp
#include <torch/extension.h>
#include <omp.h>

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv1_weight,
    torch::Tensor conv2_weight,
    torch::Tensor bn1_weight,
    torch::Tensor bn1_bias,
    torch::Tensor bn1_mean,
    torch::Tensor bn1_var,
    torch::Tensor bn2_weight,
    torch::Tensor bn2_bias,
    torch::Tensor bn2_mean,
    torch::Tensor bn2_var,
    torch::Tensor downsample_conv_weight,
    torch::Tensor downsample_bn_weight,
    torch::Tensor downsample_bn_bias,
    torch::Tensor downsample_bn_mean,
    torch::Tensor downsample_bn_var,
    int64_t stride,
    bool is_training) {

  // Save identity for residual connection
  auto identity = x.clone();

  // First conv block
  auto out = torch::conv2d(x, conv1_weight, {}, {stride, stride}, {1, 1});
  out = torch::batch_norm(out, bn1_weight, bn1_bias, bn1_mean, bn1_var, is_training, 0.1, 1e-5, true);
  out = torch::relu(out);

  // Second conv block
  out = torch::conv2d(out, conv2_weight, {}, {1, 1}, {1, 1});
  out = torch::batch_norm(out, bn2_weight, bn2_bias, bn2_mean, bn2_var, is_training, 0.1, 1e-5, true);

  // Downsample path - explicit IntArrayRef for padding
  identity = torch::conv2d(
      identity,
      downsample_conv_weight,
      {},
      {stride, stride},
      c10::IntArrayRef({0, 0})  // Explicit type disambiguation
  );
  identity = torch::batch_norm(
      identity,
      downsample_bn_weight,
      downsample_bn_bias,
      downsample_bn_mean,
      downsample_bn_var,
      is_training,
      0.1,
      1e-5,
      true
  );

  // Add and final activation
  out += identity;
  out = torch::relu(out);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "BasicBlock forward");
}