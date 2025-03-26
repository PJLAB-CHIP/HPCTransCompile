```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declaration of CUDA kernel
torch::Tensor conv_transpose3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups);

// C++ interface
torch::Tensor conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
  return conv_transpose3d_cuda_forward(input, weight, bias, stride, padding, output_padding, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose3d_forward, "ConvTranspose3d forward (CUDA)");
}
```

```cu
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_width,
    const int input_height,
    const int output_depth,
    const int output_width,
    const int output_height,
    const int kernel_depth,
    const int kernel_width,
    const int kernel_height,
    const int stride_depth,
    const int stride_width,
    const int stride_height,
    const int padding_depth,
    const int padding_width,
    const int padding_height,
    const int output_padding_depth,
    const int output_padding_width,
    const int output_padding_height,
    const int groups) {

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= batch_size * out_channels * output_depth * output_width * output_height) return;

  // Decompose the flattened index
  const int b = n / (out_channels * output_depth * output_width * output_height);
  const int oc = (n / (output_depth * output_width * output_height)) % out_channels;
  const int od = (n / (output_width * output_height)) % output_depth;
  const int ow = (n / output_height) % output_width;
  const int oh = n % output_height;

  const int g = oc / (out_channels / groups);
  const int oc_in_group = oc % (out_channels / groups);

  float value = 0.0f;

  // Iterate over input channels in the group
  for (int ic = g * (in_channels / groups); ic < (g + 1) * (in_channels / groups); ++ic) {
    // Iterate over kernel
    for (int kd = 0; kd < kernel_depth; ++kd) {
      for (int kw = 0; kw < kernel_width; ++kw) {
        for (int kh = 0; kh < kernel_height; ++kh) {
          // Calculate input position
          const int id = od + padding_depth - kd * stride_depth;
          const int iw = ow + padding_width - kw * stride_width;
          const int ih = oh + padding_height - kh * stride_height;

          // Check if input position is valid
          if (id >= 0 && id < input_depth && iw >= 0 && iw < input_width && ih >= 0 && ih < input_height) {
            const int input_idx = ((b * in_channels + ic) * input_depth + id) * input_width + iw) * input_height + ih;
            const int weight_idx = ((ic * (out_channels / groups) + oc_in_group) * kernel_depth + kd) * kernel_width + kw) * kernel_height + kh;
            value += input[input_idx] * weight[weight_idx];
          }
        }
      }
    }
  }

  // Add bias if present
  if (bias != nullptr) {
    value += bias[oc];
  }

  output[n] = value;
}

torch::Tensor conv_transpose3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  if (bias.defined()) CHECK_INPUT(bias);

  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  const int input_depth = input.size(2);
  const int input_width = input.size(3);
  const int input_height = input.size(4);

  const int out_channels = weight.size(1) * groups;
  const int kernel_depth = weight.size(2);
  const int kernel_width = weight.size(3);
  const int kernel_height = weight.size(4);

  const int stride_depth = stride[0];
  const int stride_width = stride[1];
  const int stride_height = stride[2];

  const int padding_depth = padding[0];
  const int padding_width = padding[1];
  const int padding_height = padding[2];

  const int output_padding_depth = output_padding[0];
  const int output_padding_width = output_padding[1];
  const int output_padding_height = output_padding[2];

  // Calculate output dimensions
  const int output_depth = (input_depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
  const int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;
  const int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;

  auto output = torch::zeros({batch_size, out_channels, output_depth, output_width, output_height}, input.options());

  const int threads = 1024;
  const int elements = batch_size * out_channels * output_depth * output_width * output_height;
  const int blocks = (elements + threads - 1) / threads;

  conv_transpose3d_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.defined() ? bias.data_ptr<float>() : nullptr,
      output.data_ptr<float>(),
      batch_size,
      in_channels,
      out_channels,
      input_depth,
      input_width,
      input_height,
      output_depth,
      output_width,
      output_height,
      kernel_depth,
      kernel_width,
      kernel_height,
      stride_depth,
      stride_width,
      stride_height,
      padding_depth,
      padding_width,
      padding_height,
      output_padding_depth,
      output_padding_width,
      output_padding_height,
      groups);

  return output;
}
```