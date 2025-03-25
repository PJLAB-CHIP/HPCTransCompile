#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Define the function to perform the operations
void optimized_fused_ops_cpu(
    float* output,
    const float* conv_output,
    const float* channel_bias,
    float scaling_factor,
    int batch_size,
    int channels,
    int height,
    int width) {

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Calculate the base index
                const int base_idx = (b * channels * height + h) * width + w;

                // Find the maximum value
                float max_val = -INFINITY;
                for (int c = 0; c < channels; ++c) {
                    const int idx = base_idx + c * height * width;
                    max_val = std::max(max_val, conv_output[idx]);
                }

                // Compute exponentials and sum
                float thread_sum = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    const int idx = base_idx + c * height * width;
                    float val = exp(conv_output[idx] - max_val);
                    thread_sum += val;
                }

                // Apply softmax, bias, scaling, and sigmoid
                for (int c = 0; c < channels; ++c) {
                    const int idx = base_idx + c * height * width;
                    float val = exp(conv_output[idx] - max_val) / thread_sum;
                    val += channel_bias[c];
                    val *= scaling_factor;
                    output[idx] = 1.0f / (1.0f + exp(-val));
                }
            }
        }
    }
}

// Define the forward function
torch::Tensor forward(
    torch::Tensor x,
    int stride,
    int padding,
    int output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias,
    float scaling_factor) {

    // Perform transposed convolution using PyTorch
    auto conv_out = torch::nn::functional::conv_transpose2d(
        x, conv_transpose,
        torch::nn::functional::ConvTranspose2dFuncOptions()
            .bias(conv_transpose_bias)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
    );

    TORCH_CHECK(bias.size(0) == conv_out.size(1), "Bias size must match channel dimension");

    auto output = torch::empty_like(conv_out);
    const int batch_size = conv_out.size(0);
    const int channels = conv_out.size(1);
    const int height = conv_out.size(2);
    const int width = conv_out.size(3);

    optimized_fused_ops_cpu(
        output.data_ptr<float>(),
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        batch_size,
        channels,
        height,
        width
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Fused ConvTranspose2d+Softmax+Bias+Scale+Sigmoid on CPU");
}