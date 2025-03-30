#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <omp.h>

// Device function: GELU approximation
__device__ inline float gelu(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}

// CUDA kernel that performs convolution, scalar multiplication, LeakyReLU and GELU.
// - input: [batch_size, in_channels, input_h, input_w]
// - weight: [out_channels, in_channels, kernel_size, kernel_size]
// - bias: [out_channels]
// - multiplier: [out_channels] (broadcast over spatial dims)
// - output: [batch_size, out_channels, output_h, output_w]
__global__ void conv_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ multiplier,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_h,
    int input_w,
    int out_channels,
    int kernel_size,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_h * output_w;
    
    // Grid-stride loop to cover all output elements.
    while (idx < total) {
        // Recover indices: idx corresponds to (n, oc, oh, ow)
        int ow = idx % output_w;
        int tmp = idx / output_w;
        int oh = tmp % output_h;
        tmp = tmp / output_h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        // Start with the bias for output channel oc.
        float sum = bias[oc];
        
        // Convolution: iterate over input channels and kernel window.
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int in_h = oh + i; // stride = 1, no padding.
                    int in_w = ow + j;
                    int input_index = ((n * in_channels + ic) * input_h + in_h) * input_w + in_w;
                    int weight_index = ((oc * in_channels + ic) * kernel_size + i) * kernel_size + j;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
        
        // Multiply with the channel-specific multiplier.
        sum *= multiplier[oc];
        
        // Apply LeakyReLU activation (negative slope = 0.01).
        sum = (sum > 0.0f) ? sum : 0.01f * sum;
        
        // Apply GELU activation.
        float out_val = gelu(sum);
        
        output[idx] = out_val;
        idx += blockDim.x * gridDim.x;
    }
}

// C++ interface (to be called from Python)
torch::Tensor forward_cpu(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor multiplier
) {
    // Get input dimensions.
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_h = input.size(2);
    const auto input_w = input.size(3);
    
    // Get convolution parameters.
    const auto out_channels = conv_weight.size(0);
    const auto kernel_size = conv_weight.size(2);
    const auto output_h = input_h - kernel_size + 1;
    const auto output_w = input_w - kernel_size + 1;
    
    // Allocate output tensor.
    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());
    
    // Launch CPU kernel (using OpenMP for parallelism)
    const int total_elements = batch_size * out_channels * output_h * output_w;
    const int threads = 256;
    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; idx++) {
        // Recover indices: idx corresponds to (n, oc, oh, ow)
        int ow = idx % output_w;
        int tmp = idx / output_w;
        int oh = tmp % output_h;
        tmp = tmp / output_h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;

        // Start with the bias for output channel oc.
        float sum = bias[oc];
        
        // Convolution: iterate over input channels and kernel window.
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int in_h = oh + i; // stride = 1, no padding.
                    int in_w = ow + j;
                    int input_index = ((n * in_channels + ic) * input_h + in_h) * input_w + in_w;
                    int weight_index = ((oc * in_channels + ic) * kernel_size + i) * kernel_size + j;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
        
        // Multiply with the channel-specific multiplier.
        sum *= multiplier[oc];
        
        // Apply LeakyReLU activation (negative slope = 0.01).
        sum = (sum > 0.0f) ? sum : 0.01f * sum;
        
        // Apply GELU activation.
        float out_val = gelu(sum);
        
        output[idx] = out_val;
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Convolution, scalar multiplication, LeakyReLU and GELU (CPU)");
}