#include <torch/extension.h> 
#include <vector>

// CUDA kernel for 3D transposed convolution
__global__ void conv_transpose3d_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    int batch_size, int in_channels, int out_channels, int input_depth,
    int input_height, int input_width, int output_depth, int output_height,
    int output_width, int kernel_size, int stride, int padding, int dilation) {
  // Compute global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements =
      batch_size * out_channels * output_depth * output_height * output_width;

  // Return if thread index exceeds output size
  if (idx >= total_elements)
    return;

  // Compute output indices (n, c, do, ho, wo) from thread index
  int wo = idx % output_width;
  idx /= output_width;
  int ho = idx % output_height;
  idx /= output_height;
  int do_ = idx % output_depth; // Use do_ to avoid C++ keyword 'do'
  idx /= output_depth;
  int c = idx % out_channels;
  int n = idx / out_channels;

  // Initialize output value
  float sum = 0.0f;

  // Loop over input channels and kernel dimensions
  for (int k = 0; k < in_channels; k++) {
    for (int kd = 0; kd < kernel_size; kd++) {
      int di = do_ * stride - padding + kd * dilation;
      if (di >= 0 && di < input_depth) {
        for (int kh = 0; kh < kernel_size; kh++) {
          int hi = ho * stride - padding + kh * dilation;
          if (hi >= 0 && hi < input_height) {
            for (int kw = 0; kw < kernel_size; kw++) {
              int wi = wo * stride - padding + kw * dilation;
              if (wi >= 0 && wi < input_width) {
                // Compute input and weight offsets for contiguous tensors
                int input_offset =
                    n * (in_channels * input_depth * input_height *
                         input_width) +
                    k * (input_depth * input_height * input_width) +
                    di * (input_height * input_width) + hi * input_width + wi;
                int weight_offset =
                    k * (out_channels * kernel_size * kernel_size *
                         kernel_size) +
                    c * (kernel_size * kernel_size * kernel_size) +
                    kd * (kernel_size * kernel_size) + kh * kernel_size + kw;
                sum += input[input_offset] * weight[weight_offset];
              }
            }
          }
        }
      }
    }
  }

  // Compute output offset
  int output_offset =
      n * (out_channels * output_depth * output_height * output_width) +
      c * (output_depth * output_height * output_width) +
      do_ * (output_height * output_width) + ho * output_width + wo;

  // Add bias if present
  if (bias != nullptr) {
    sum += bias[c];
  }

  // Write result to output
  output[output_offset] = sum;
}

// Host function to manage kernel launch
torch::Tensor conv_transpose3d_forward(torch::Tensor input,
                                       torch::Tensor weight, torch::Tensor bias,
                                       int stride, int padding, int dilation) {
  // Extract input dimensions
  auto batch_size = input.size(0);
  auto in_channels = input.size(1);
  auto input_depth = input.size(2);
  auto input_height = input.size(3);
  auto input_width = input.size(4);

  // Extract weight dimensions
  auto out_channels = weight.size(1);
  auto kernel_size =
      weight.size(2); // Assumes kernel_size is uniform across D, H, W

  // Compute output dimensions using PyTorch's transposed convolution formula
  auto output_depth = (input_depth - 1) * stride - 2 * padding +
                      dilation * (kernel_size - 1) + 1;
  auto output_height = (input_height - 1) * stride - 2 * padding +
                       dilation * (kernel_size - 1) + 1;
  auto output_width = (input_width - 1) * stride - 2 * padding +
                      dilation * (kernel_size - 1) + 1;

  // Allocate output tensor with the same options (device, dtype) as input
  auto output = torch::empty(
      {batch_size, out_channels, output_depth, output_height, output_width},
      input.options());

  // Calculate total number of output elements
  int total_elements =
      batch_size * out_channels * output_depth * output_height * output_width;

  // Define block and grid sizes
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  // Launch CUDA kernel
  conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
      input.data_ptr<float>(), weight.data_ptr<float>(),
      bias.defined() ? bias.data_ptr<float>() : nullptr,
      output.data_ptr<float>(), batch_size, in_channels, out_channels,
      input_depth, input_height, input_width, output_depth, output_height,
      output_width, kernel_size, stride, padding, dilation);

  return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose3d_forward,
        "Conv Transpose 3D forward (CUDA)");
}