#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define UNROLL_BLOCK_SIZE 256

template<int KERNEL_H, int KERNEL_W>
__global__ void unrolled_fused_conv_instnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    float* __restrict__ output,
    const float* __restrict__ inst_scale,
    const float* __restrict__ inst_shift,
    float divide_by,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int output_height,
    int output_width,
    float epsilon) {

  int n = blockIdx.x;
  int oc = blockIdx.y;
  int tid = threadIdx.x;
  int num_pixels = output_height * output_width;
  
  int out_base = ((n * out_channels + oc) * output_height) * output_width;
  
  float local_sum = 0.0f;
  float local_sum_sq = 0.0f;

  // Grid-stride loop over output pixels
  for (int idx = tid; idx < num_pixels; idx += blockDim.x) {
    int w_out = idx % output_width;
    int h_out = idx / output_width;
    float conv_val = conv_bias[oc];

    // Unroll channel loop for better instruction-level parallelism
    #pragma unroll 4
    for (int ic = 0; ic < in_channels; ++ic) {
      // Manual unroll for small kernel sizes (3x3 or 5x5 typically)
      #pragma unroll
      for (int i = 0; i < KERNEL_H; ++i) {
        #pragma unroll
        for (int j = 0; j < KERNEL_W; ++j) {
          int in_h = h_out + i;
          int in_w = w_out + j;
          int input_idx = ((n * in_channels + ic) * input_height + in_h) * input_width + in_w;
          int weight_idx = ((oc * in_channels + ic) * KERNEL_H + i) * KERNEL_W + j;
          conv_val = __fmaf_rn(input[input_idx], conv_weight[weight_idx], conv_val);
        }
      }
    }

    output[out_base + idx] = conv_val;
    local_sum += conv_val;
    local_sum_sq += conv_val * conv_val;
  }

  // Warp-level reduction using shuffle instructions
  unsigned int mask = 0xffffffff;
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  // Block-level reduction using shared memory
  int num_warps = blockDim.x / 32;
  extern __shared__ float shared[];
  float* warp_sum = shared;
  float* warp_sum_sq = shared + num_warps;
  float* stats = shared + 2 * num_warps;

  int laneId = tid & 31;
  int warpId = tid >> 5;
  
  if (laneId == 0) {
    warp_sum[warpId] = local_sum;
    warp_sum_sq[warpId] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction in the first warp
  if (tid < 32) {
    float block_sum = (tid < num_warps) ? warp_sum[tid] : 0.0f;
    float block_sum_sq = (tid < num_warps) ? warp_sum_sq[tid] : 0.0f;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      block_sum += __shfl_down_sync(mask, block_sum, offset);
      block_sum_sq += __shfl_down_sync(mask, block_sum_sq, offset);
    }

    if (tid == 0) {
      float mean = block_sum / num_pixels;
      float variance = block_sum_sq / num_pixels - mean * mean;
      stats[0] = mean;
      stats[1] = rsqrtf(variance + epsilon);
    }
  }
  __syncthreads();

  float mean = stats[0];
  float inv_std = stats[1];
  float scale = inst_scale[oc];
  float shift = inst_shift[oc];

  // Normalize and scale output values
  #pragma unroll 4
  for (int idx = tid; idx < num_pixels; idx += blockDim.x) {
    int out_idx = out_base + idx;
    float val = output[out_idx];
    float norm = (val - mean) * inv_std;
    output[out_idx] = (scale * norm + shift) / divide_by;
  }
}

torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    c10::optional<torch::Tensor> inst_scale_opt,
    c10::optional<torch::Tensor> inst_shift_opt,
    float divide_by) {

  input = input.contiguous();
  conv_weight = conv_weight.contiguous();
  conv_bias = conv_bias.contiguous();

  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);
  int out_channels = conv_weight.size(0);
  int kernel_h = conv_weight.size(2);
  int kernel_w = conv_weight.size(3);

  int output_height = input_height - kernel_h + 1;
  int output_width = input_width - kernel_w + 1;

  auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

  torch::Tensor inst_scale, inst_shift;
  if (inst_scale_opt.has_value() && inst_scale_opt.value().defined()) {
    inst_scale = inst_scale_opt.value().contiguous();
  } else {
    inst_scale = torch::ones({out_channels}, output.options());
  }
  if (inst_shift_opt.has_value() && inst_shift_opt.value().defined()) {
    inst_shift = inst_shift_opt.value().contiguous();
  } else {
    inst_shift = torch::zeros({out_channels}, output.options());
  }

  dim3 grid(batch_size, out_channels);
  int threads = UNROLL_BLOCK_SIZE;
  int num_warps = threads / 32;
  size_t shared_mem_bytes = (2 * num_warps + 2) * sizeof(float);

  float epsilon = 1e-5f;

  if (kernel_h == 3 && kernel_w == 3) {
    unrolled_fused_conv_instnorm_kernel<3, 3><<<grid, threads, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        inst_scale.data_ptr<float>(),
        inst_shift.data_ptr<float>(),
        divide_by,
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        epsilon);
  } else if (kernel_h == 5 && kernel_w == 5) {
    unrolled_fused_conv_instnorm_kernel<5, 5><<<grid, threads, shared_mem_bytes>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        inst_scale.data_ptr<float>(),
        inst_shift.data_ptr<float>(),
        divide_by,
        batch_size,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        epsilon);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in unrolled_fused_conv_instnorm_kernel: %s\n", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cuda, "Unrolled Fused Conv2d + InstanceNorm + Division (CUDA)");
}