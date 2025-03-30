#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define THREADS_PER_BLOCK 256

// Forward function to invoke the CPU kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float subtract_value_1,
    float subtract_value_2) {

    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(conv_weight.is_cpu(), "conv_weight must be a CPU tensor");
    TORCH_CHECK(conv_bias.is_cpu(), "conv_bias must be a CPU tensor");

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

    #pragma omp parallel for
    for (int index = 0; index < total_elements; index++) {
        // Decode the flat index into (n, oc, oh, ow)
        int ow = index % out_w;
        int tmp = index / out_w;
        int oh = tmp % out_h;
        tmp /= out_h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;
        
        float sum = conv_bias[oc];

        // Special case: if kernel_size == 3, manually unroll the convolution loops
        if (kernel_size == 3) {
            for (int ic = 0; ic < in_channels; ic++) {
                int base_input = n * (in_channels * in_h * in_w) + ic * (in_h * in_w);
                int base_weight = oc * (in_channels * 9) + ic * 9;  // 3x3 kernel ==> 9 elements
                
                int offset = oh * in_w + ow;
                sum += static_cast<float>(input[base_input + offset])     * static_cast<float>(weight[base_weight]);
                sum += static_cast<float>(input[base_input + offset + 1]) * static_cast<float>(weight[base_weight + 1]);
                sum += static_cast<float>(input[base_input + offset + 2]) * static_cast<float>(weight[base_weight + 2]);
                
                offset = (oh + 1) * in_w + ow;
                sum += static_cast<float>(input[base_input + offset])     * static_cast<float>(weight[base_weight + 3]);
                sum += static_cast<float>(input[base_input + offset + 1]) * static_cast<float>(weight[base_weight + 4]);
                sum += static_cast<float>(input[base_input + offset + 2]) * static_cast<float>(weight[base_weight + 5]);
                
                offset = (oh + 2) * in_w + ow;
                sum += static_cast<float>(input[base_input + offset])     * static_cast<float>(weight[base_weight + 6]);
                sum += static_cast<float>(input[base_input + offset + 1]) * static_cast<float>(weight[base_weight + 7]);
                sum += static_cast<float>(input[base_input + offset + 2]) * static_cast<float>(weight[base_weight + 8]);
            }
        } else {
            // Fallback: use loop unrolling for general kernel sizes
            #pragma omp parallel for
            for (int ic = 0; ic < in_channels; ic++) {
                #pragma omp parallel for
                for (int kh = 0; kh < kernel_size; kh++) {
                    #pragma omp parallel for
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        sum += static_cast<float>(input[input_idx]) * static_cast<float>(weight[weight_idx]);
                    }
                }
            }
        }
        
        // Apply the subtraction values
        sum = sum - subtract_value_1 - subtract_value_2;
        
        // Apply Mish activation: mish(x) = x * tanh( log(1 + exp(x)) )
        float softplus = logf(1.0f + expf(sum));
        float mish = sum * tanhf(softplus);
        
        output[index] = mish;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution, subtract two values, and apply Mish activation (CPU) with manual unrolling for kernel_size == 3");
}