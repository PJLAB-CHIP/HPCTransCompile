#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cmath>

#define THREADS_PER_BLOCK 256

// CUDA kernel that performs a 2D convolution (stride=1, no padding), subtracts two scalar values,
// and applies the Mish activation function. For the common case of a 3x3 kernel,
// the inner loops are manually unrolled to reduce loop overhead and improve performance.
__global__ void conv_mish_manual_unroll_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float subtract1,
    float subtract2,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w) {

    int total_elements = batch_size * out_channels * out_h * out_w;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_elements; index += blockDim.x * gridDim.x) {
        // Decode the flat index into (n, oc, oh, ow)
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp /= out_h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;
        
        float sum = bias[oc];

        // Special case: if kernel_size == 3, manually unroll the convolution loops
        if (kernel_size == 3) {
            for (int ic = 0; ic < in_channels; ic++) {
                int base_input = n * (in_channels * in_h * in_w) + ic * (in_h * in_w);
                int base_weight = oc * (in_channels * 9) + ic * 9;  // 3x3 kernel ==> 9 elements
                
                int offset = oh * in_w + ow;
                sum += input[base_input + offset]     * weight[base_weight];
                sum += input[base_input + offset + 1] * weight[base_weight + 1];
                sum += input[base_input + offset + 2] * weight[base_weight + 2];
                
                offset = (oh + 1) * in_w + ow;
                sum += input[base_input + offset]     * weight[base_weight + 3];
                sum += input[base_input + offset + 1] * weight[base_weight + 4];
                sum += input[base_input + offset + 2] * weight[base_weight + 5];
                
                offset = (oh + 2) * in_w + ow;
                sum += input[base_input + offset]     * weight[base_weight + 6];
                sum += input[base_input + offset + 1] * weight[base_weight + 7];
                sum += input[base_input + offset + 2] * weight[base_weight + 8];
            }
        } else {
            // Fallback: use loop unrolling for general kernel sizes
            #pragma unroll
            for (int ic = 0; ic < in_channels; ic++) {
                #pragma unroll
                for (int kh = 0; kh < kernel_size; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Apply the subtraction values
        sum = sum - subtract1 - subtract2;
        
        // Apply Mish activation: mish(x) = x * tanh( log(1 + exp(x)) )
        float softplus = logf(1.0f + expf(sum));
        float mish = sum * tanhf(softplus);
        
        output[index] = mish;
    }
}


// Forward function to invoke the CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float subtract_value_1,
    float subtract_value_2) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");

    x = x.contiguous();
    conv_weight = conv_weight.contiguous();
    conv_bias = conv_bias.contiguous();

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    
    // conv_weight shape: (out_channels, in_channels, kernel_size, kernel_size)
    int out_channels = conv_weight.size(0);
    int kernel_size = conv_weight.size(2);  // assuming square kernel
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    int total_elements = batch_size * out_channels * out_h * out_w;
    int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_mish_manual_unroll_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        x.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        subtract_value_1,
        subtract_value_2,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_size,
        out_h,
        out_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution, subtract two values, and apply Mish activation (CUDA) with manual unrolling for kernel_size == 3");
}
