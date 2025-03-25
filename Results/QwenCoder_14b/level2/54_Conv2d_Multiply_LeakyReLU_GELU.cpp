#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// CPU function: GELU approximation
inline float gelu(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}

// CPU function that performs convolution, scalar multiplication, LeakyReLU and GELU.
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

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    // Start with the bias for output channel oc.
                    float sum = conv_bias[oc];
                    
                    // Convolution: iterate over input channels and kernel window.
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int i = 0; i < kernel_size; ++i) {
                            for (int j = 0; j < kernel_size; ++j) {
                                int in_h = oh + i; // stride = 1, no padding.
                                int in_w = ow + j;
                                int input_index = ((n * in_channels + ic) * input_h + in_h) * input_w + in_w;
                                int weight_index = ((oc * in_channels + ic) * kernel_size + i) * kernel_size + j;
                                sum += input[input_index] * conv_weight[weight_index];
                            }
                        }
                    }
                    
                    // Multiply with the channel-specific multiplier.
                    sum *= multiplier[oc];
                    
                    // Apply LeakyReLU activation (negative slope = 0.01).
                    sum = (sum > 0.0f) ? sum : 0.01f * sum;
                    
                    // Apply GELU activation.
                    float out_val = gelu(sum);
                    
                    output[n][oc][oh][ow] = out_val;
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Convolution, scalar multiplication, LeakyReLU and GELU (CPU)");
}