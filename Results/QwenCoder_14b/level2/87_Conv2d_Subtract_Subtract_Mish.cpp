#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Function to perform the convolution, subtraction, and Mish activation
void conv_mish_manual_unroll_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float subtract1,
    float subtract2,
    float* output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w) {

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < batch_size; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
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

                    output[n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow] = mish;
                }
            }
        }
    }
}

// Forward function to invoke the CPU function
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float subtract_value_1,
    float subtract_value_2) {

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

    conv_mish_manual_unroll_cpu(
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
    m.def("forward", &forward, "Convolution, subtract two values, and apply Mish activation (CPU) with manual unrolling for kernel_size == 3");
}