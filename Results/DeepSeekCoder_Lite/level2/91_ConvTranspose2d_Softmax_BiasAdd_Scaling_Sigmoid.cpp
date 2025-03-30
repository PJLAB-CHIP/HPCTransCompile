#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Choose a BLOCK_SIZE that is optimal for the target hardware
#define BLOCK_SIZE 128

template <typename scalar_t>
void optimized_fused_ops_kernel_minimized_warp_divergence(
    scalar_t* output,
    const scalar_t* conv_output,
    const scalar_t* channel_bias,
    float scaling_factor,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    const int spatial_size = height * width;
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int spatial_pos = b * spatial_size + h * width + w;
                scalar_t thread_max = -INFINITY;
                scalar_t thread_sum = 0;

                // Pre-compute base index to avoid redundant calculations
                int base_idx = (b * channels * height + h) * width + w;

                // First pass: Find maximum while prefetching data
                for (int c = 0; c < channels; ++c) {
                    int idx = base_idx + c * height * width;
                    scalar_t val = conv_output[idx];
                    thread_max = std::max(thread_max, val);
                }

                // Reduce maximum within thread
                for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
                    if (omp_get_thread_num() % (2 * stride) == 0) {
                        thread_max = std::max(thread_max, thread_max + stride);
                    }
                    #pragma omp barrier
                }

                const scalar_t max_val = thread_max;

                // Second pass: Compute exponentials and sum
                for (int c = 0; c < channels; ++c) {
                    int idx = base_idx + c * height * width;
                    scalar_t val = std::exp(conv_output[idx] - max_val);
                    thread_sum += val;
                }

                // Final pass: Apply softmax, bias, scaling, and sigmoid
                for (int c = 0; c < channels; ++c) {
                    int idx = base_idx + c * height * width;
                    scalar_t val = std::exp(conv_output[idx] - max_val) / thread_sum;
                    val += channel_bias[c];
                    val *= scaling_factor;
                    output[idx] = 1.0f / (1.0f + std::exp(-val));
                }
            }
        }
    }
}

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

    const int total_spatial = batch_size * height * width;
    const int blocks = (total_spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(conv_out.scalar_type(), "optimized_fused_ops_kernel_minimized_warp_divergence", ([&] {
        optimized_fused_ops_kernel_minimized_warp_divergence<scalar_t>(
            output.data_ptr<scalar_t>(),
            conv_out.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            scaling_factor,
            batch_size,
            channels,
            height,
            width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Fused ConvTranspose2d+Softmax+Bias+Scale+Sigmoid with minimized warp divergence");
}